# pip install playwright
# playwright install chromium

from __future__ import annotations

import os
import sys
import asyncio
import random
import subprocess
import tempfile
from typing import Optional

from logger import logger

from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError

REELS_URL = "https://www.instagram.com/reels/"
SESSION_FILE = "ig_session.json"


class Reel:
    """
    Represents a single Instagram reel.

    States:
      - peeked:  known URL, not yet on-screen. seek_next() returns this.
      - active:  currently visible in the Instagram tab.

    The Instagram tab stays in the background (idle tab is in front) until
    play() or like() is called. After play() finishes, the idle tab is
    brought back to front so Instagram stops counting watch time.

    download() and seek_next() never touch tab focus — safe to call while
    Instagram is backgrounded.
    """

    def __init__(self, url: str, ig_page: Page, idle_page: Page, context: BrowserContext):
        self.url = url
        self._ig_page = ig_page
        self._idle_page = idle_page
        self._context = context

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def play(self, t: float) -> None:
        """
        Bring the Instagram tab to front, scroll to this reel if needed,
        watch for t seconds, then return focus to the idle tab.
        """
        await self._ig_page.bring_to_front()
        await self._ensure_active()
        await self._ig_page.wait_for_timeout(t * 1000)
        await self._idle_page.bring_to_front()

    async def like(self) -> bool:
        """
        Like this reel. Brings Instagram tab to front, clicks like,
        then returns to idle tab.
        Returns True if the like button was clicked, False otherwise.
        """
        await self._ig_page.bring_to_front()
        await self._ensure_active()

        # Find the viewport coords of the like button for the on-screen reel.
        # JS returns the center point; Playwright then physically moves the
        # mouse there and clicks — avoids the DOM-order issue with .click().
        pos = await self._ig_page.evaluate("""() => {
            const selectors = [
                'svg[aria-label="Like"]',
                'svg[aria-label="like"]',
                'button[aria-label="Like"]',
            ];
            const vh = window.innerHeight;
            const vw = window.innerWidth;
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    const r = el.getBoundingClientRect();
                    const cy = (r.top + r.bottom) / 2;
                    const cx = (r.left + r.right) / 2;
                    if (cy >= 0 && cy <= vh && cx >= 0 && cx <= vw) {
                        return { x: cx, y: cy };
                    }
                }
            }
            return null;
        }""")

        liked = False
        if pos:
            await self._ig_page.mouse.move(pos["x"], pos["y"])
            await self._ig_page.wait_for_timeout(200)
            await self._ig_page.mouse.click(pos["x"], pos["y"])
            liked = True

        return liked

    async def download(self, save_path: str) -> str:
        """
        Download this reel to save_path using yt-dlp, authenticated via
        the current browser session's cookies.

        Does NOT require the reel to be active and does not affect tab
        focus — safe to call while Instagram is backgrounded.

        Returns save_path on success.
        Raises RuntimeError if yt-dlp exits non-zero.
        """
        cookies = await self._context.cookies()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Netscape HTTP Cookie File\n")
            for c in cookies:
                domain = c["domain"]
                flag = "TRUE" if domain.startswith(".") else "FALSE"
                path = c.get("path", "/")
                secure = "TRUE" if c.get("secure") else "FALSE"
                expires = int(c.get("expires") or 0)
                f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{c['name']}\t{c['value']}\n")
            cookie_file = f.name

        output_dir = os.path.dirname(save_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        try:
            cmd = [
                self.find_ytdlp(),
                "-f", "b",
                "--recode-video", "mp4",
                "-o", save_path,
                "--retries", "3",
                "--fragment-retries", "3",
                "--write-info-json",
                "--cookies", cookie_file,
                self.url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")
        finally:
            os.unlink(cookie_file)

        return save_path
    
    def find_ytdlp(self) -> Optional[str]:
        """Find yt-dlp executable in common locations."""
        possible_commands = [
            "yt-dlp",
            "yt-dlp.exe",
            os.path.join(sys.prefix, "Scripts", "yt-dlp.exe"),
            os.path.join(os.path.dirname(sys.executable), "Scripts", "yt-dlp.exe"),
        ]

        for cmd in possible_commands:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    async def seek_next(self) -> Optional["Reel"]:
        """
        Get the URL of the next reel by scrolling down, capturing the URL,
        scrolling back up, then returning to the idle tab.

        Returns a Reel for the next reel (peeked, not active), or None.
        """
        await self._ig_page.bring_to_front()

        await self._ig_page.keyboard.press("ArrowDown")
        logger.debug("Scrolled down to peek at next reel.")
        await self._ig_page.wait_for_timeout(2000)
        next_url = self._ig_page.url

        await self._idle_page.bring_to_front()

        if "/reels/" not in next_url or self._norm(next_url) == self._norm(self.url):
            return None

        return Reel(next_url, self._ig_page, self._idle_page, self._context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_active(self) -> None:
        """
        Scroll the Instagram tab to this reel if it isn't on-screen yet.
        Assumes the Instagram tab is already in the foreground.
        """
        if self._norm(self._ig_page.url) == self._norm(self.url):
            return

        raise RuntimeError(f"Could not scroll to reel: {self.url}")

    @staticmethod
    def _norm(url: str) -> str:
        if url.startswith("/"):
            url = "https://www.instagram.com" + url
        return url.split("?")[0].rstrip("/") + "/"

    def __repr__(self) -> str:
        return f"Reel('{self.url}')"


# ----------------------------------------------------------------------
# ReelsAgent — browser lifecycle + entry point
# ----------------------------------------------------------------------

class ReelsAgent:
    """
    Manages the browser session. Keeps an idle tab in front by default so
    Instagram doesn't accumulate watch time between decisions.

    Usage:
        async with ReelsAgent() as agent:
            reel = await agent.current_reel()

            next_reel = await reel.seek_next()            # peek, no focus change
            path = await next_reel.download("/tmp/r.mp4") # download, no focus change
            # run classifier on path ...
            await next_reel.play(8)   # IG tab comes front for 8s, then goes back
            await next_reel.like()    # same — front briefly, then back to idle
    """

    def __init__(self, headless: bool = False, session_file: str = SESSION_FILE):
        self.headless = headless
        self.session_file = session_file
        self._playwright = None
        self._browser = None
        self._context: Optional[BrowserContext] = None
        self._ig_page: Optional[Page] = None
        self._idle_page: Optional[Page] = None

    async def __aenter__(self) -> "ReelsAgent":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(storage_state=self.session_file)

        # Load Instagram reels in one tab
        self._ig_page = await self._context.new_page()
        await self._ig_page.goto(REELS_URL, timeout=20000)
        await self._ig_page.locator("video").first.wait_for(state="visible", timeout=15000)

        # Open a blank idle tab and bring it to front — IG goes to background
        self._idle_page = await self._context.new_page()
        await self._idle_page.goto("about:blank")
        await self._idle_page.bring_to_front()

        return self

    async def __aexit__(self, *_) -> None:
        await self._browser.close()
        await self._playwright.stop()

    async def current_reel(self) -> Reel:
        """Return a Reel for the reel currently loaded in the Instagram tab."""
        return Reel(self._ig_page.url, self._ig_page, self._idle_page, self._context)
    
    async def scroll_to_next(self) -> Reel:
        """Scroll the Instagram tab to the next reel and return a Reel for it."""
        await self._ig_page.bring_to_front()
        await self._ig_page.keyboard.press("ArrowDown")
        await self._ig_page.wait_for_timeout(1000)
        return Reel(self._ig_page.url, self._ig_page, self._idle_page, self._context)


# ----------------------------------------------------------------------
# Session helper — run once to create ig_session.json
# ----------------------------------------------------------------------

async def save_session():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # IMPORTANT: visible browser
        context = await browser.new_context()
        page = await context.new_page()

        # Go to Instagram login
        await page.goto("https://www.instagram.com/accounts/login/")

        logger.info("Please log in to Instagram in the opened browser window.")
        input("Press ENTER after logging in successfully...")

        # Save session
        await context.storage_state(path="ig_session.json")
        logger.debug(f"Session saved to ig_session.json")

        await browser.close()

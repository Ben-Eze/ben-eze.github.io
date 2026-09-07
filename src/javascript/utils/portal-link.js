/*
 * portal-link.js
 *
 * Give any <a class="portal-link"> wrapping an <img> or <video> a zoom-through
 * transition: the media is flown at the viewer while the page fades out behind
 * it, then the link is followed.
 */

(function () {
  "use strict";

  const DURATION = 520;   // keep in step with the transition in page.css

  /* A snapshot of the element as it looks right now. Cloning the node would
     work for an <img>, but a cloned <video> starts paused at its first frame,
     which reads as a jump cut the moment the animation begins. */
  function snapshot(media, rect) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));

    try {
      canvas.getContext("2d").drawImage(media, 0, 0, canvas.width, canvas.height);
    } catch (error) {
      // A video that hasn't buffered a frame yet, or a tainted canvas
      console.warn("portal-link: could not snapshot", error);
      return null;
    }
    return canvas;
  }

  function fly(link, media) {
    const rect = media.getBoundingClientRect();
    const canvas = snapshot(media, rect);
    if (!canvas) {
      return false;
    }

    canvas.className = "portal-flight";
    canvas.style.left = `${rect.left}px`;
    canvas.style.top = `${rect.top}px`;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const backdrop = document.createElement("div");
    backdrop.className = "portal-backdrop";

    document.body.append(backdrop, canvas);

    // Force the starting geometry to be computed. Without this the browser
    // folds the append and the change below into a single style pass, sees
    // only the end state, and skips the transition entirely
    void canvas.offsetWidth;

    // Enough to cover the viewport however the media is proportioned
    const scale = Math.max(
      window.innerWidth / rect.width,
      window.innerHeight / rect.height
    ) * 1.1;
    const dx = window.innerWidth / 2 - (rect.left + rect.width / 2);
    const dy = window.innerHeight / 2 - (rect.top + rect.height / 2);

    backdrop.classList.add("is-visible");
    canvas.style.transform = `translate(${dx}px, ${dy}px) scale(${scale})`;
    canvas.style.borderRadius = "0";
    canvas.style.opacity = "0";
    return true;
  }

  function bind(link) {
    const media = link.querySelector("img, video");
    if (!media) {
      return;
    }

    link.addEventListener("click", function (event) {
      // Leave ctrl/cmd/shift-clicks and middle-clicks alone, so opening in a
      // new tab still works
      if (event.button !== 0 || event.metaKey || event.ctrlKey
        || event.shiftKey || event.altKey) {
        return;
      }
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        return;
      }

      event.preventDefault();

      if (!fly(link, media)) {
        window.location.href = link.href;   // snapshot failed, just go
        return;
      }

      setTimeout(function () {
        window.location.href = link.href;
      }, DURATION);
    });
  }

  function init() {
    document.querySelectorAll("a.portal-link").forEach(bind);
  }

  /* Going back restores this page from the back/forward cache with its DOM
     exactly as it was left - mid-transition, with the backdrop still covering
     everything - so the leftovers have to be cleared out. pageshow fires on
     both a fresh load and a bfcache restore, and removing nothing is harmless. */
  window.addEventListener("pageshow", function () {
    document.querySelectorAll(".portal-backdrop, .portal-flight")
      .forEach(function (node) {
        node.remove();
      });

    // A restored page also brings its <video> back paused, since autoplay
    // only fires on a fresh load. These are muted, so play() is allowed, but
    // it still rejects if the browser decides otherwise
    document.querySelectorAll("a.portal-link video").forEach(function (video) {
      if (!video.paused) {
        return;
      }
      const playing = video.play();
      if (playing && playing.catch) {
        playing.catch(function () {});
      }
    });
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

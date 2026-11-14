// Progressive enhancement only: every page also works without JavaScript.
(function () {
  "use strict";

  // Theme toggle (remembered per browser; storage may be unavailable, so guard it).
  var root = document.documentElement;
  try { var saved = localStorage.getItem("theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
  var toggle = document.getElementById("theme-toggle");
  if (toggle) toggle.addEventListener("click", function () {
    var cur = root.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = cur === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });

  // Species chips submit the search form immediately.
  document.querySelectorAll("form.search input[type=radio]").forEach(function (r) {
    r.addEventListener("change", function () { r.form.submit(); });
  });

  // Upload form: drag & drop, preview, description counter, inline errors.
  var form = document.getElementById("post-form");
  if (form) {
    var zone = document.getElementById("dropzone");
    var input = form.elements.file;
    var copy = zone.querySelector(".copy");
    var preview = null;
    function show(file) {
      if (!file) return;
      if (preview) preview.remove();
      var url = URL.createObjectURL(file);
      preview = document.createElement(file.type.startsWith("video") ? "video" : "img");
      preview.className = "preview"; preview.src = url;
      if (preview.tagName === "VIDEO") { preview.muted = true; preview.loop = true; preview.autoplay = true; preview.playsInline = true; }
      preview.alt = "";
      zone.prepend(preview); zone.classList.add("has-file");
      copy.textContent = file.name;
    }
    input.addEventListener("change", function () { show(input.files[0]); });
    ["dragenter", "dragover"].forEach(function (ev) { zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (ev) { zone.addEventListener(ev, function () { zone.classList.remove("over"); }); });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(input.files[0]); }
    });
    var desc = form.elements.description, counter = document.getElementById("desc-count");
    if (desc && counter) desc.addEventListener("input", function () { counter.textContent = desc.value.length + " / 2000"; });

    var errBox = document.getElementById("form-error"), btn = document.getElementById("submit-btn");
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      errBox.hidden = true; btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = btn.dataset.sending;
      fetch("/api/posts", { method: "POST", body: new FormData(form) })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
        .then(function (res) {
          if (res.ok) { location.href = "/post/" + res.j.id; return; }
          throw new Error(typeof res.j.detail === "string" ? res.j.detail : errBox.dataset.generic);
        })
        .catch(function (err) {
          errBox.textContent = err.message || errBox.dataset.generic; errBox.hidden = false;
          btn.disabled = false; btn.textContent = btn.dataset.label;
        });
    });
  }

  // Post page: while the AI check is running, poll and reload when it finishes.
  var pending = document.getElementById("pending");
  if (pending) {
    var id = pending.dataset.id;
    (function poll() {
      fetch("/api/posts/" + id).then(function (r) { return r.json(); }).then(function (j) {
        if (j.status !== "pending") location.reload(); else setTimeout(poll, 1500);
      }).catch(function () { setTimeout(poll, 3000); });
    })();
  }
})();

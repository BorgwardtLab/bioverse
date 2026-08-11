/* Force light mode so the warm brand palette is always shown. */
(function () {
  localStorage.setItem("darkMode", "light");
  document.documentElement.classList.remove("dark");
})();

window.addEventListener("DOMContentLoaded", () => {
  localStorage.setItem("darkMode", "light");
  document.documentElement.classList.remove("dark");

  const themeSwitcher = document.querySelector(
    '[aria-label="Color theme switcher"]'
  );
  if (themeSwitcher) {
    themeSwitcher.style.display = "none";
  }
});

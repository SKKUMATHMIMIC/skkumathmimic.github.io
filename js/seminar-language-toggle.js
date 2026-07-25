const languageTextKey = {
  ko: "languageTextKo",
  en: "languageTextEn",
};

const languageLabelKey = {
  ko: "languageLabelKo",
  en: "languageLabelEn",
};

const languageStorageKey = "siteLanguage";

const saveLanguage = (language) => {
  try {
    localStorage.setItem(languageStorageKey, language);
  } catch (error) {
    // Ignore localStorage access errors.
  }
};

const getSavedLanguage = () => {
  try {
    return localStorage.getItem(languageStorageKey);
  } catch (error) {
    return null;
  }
};

const applyLanguage = (target) => {
  const language = target === "en" ? "en" : "ko";

  document.documentElement.lang = language;
  document.documentElement.dataset.language = language;
  saveLanguage(language);

  document.querySelectorAll("[data-language-toggle]").forEach((toggleRoot) => {
    toggleRoot.querySelectorAll("[data-language-button]").forEach((button) => {
      const isActive = button.dataset.languageButton === language;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));

      const labelKey = languageLabelKey[language];
      const localizedLabel = button.dataset[labelKey];
      if (typeof localizedLabel === "string" && localizedLabel !== "") {
        button.textContent = localizedLabel;
      }
    });
  });

  document.querySelectorAll("[data-language-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.languagePanel !== language;
  });

  const textKey = languageTextKey[language];
  document.querySelectorAll("[data-language-text]").forEach((node) => {
    const localizedText = node.dataset[textKey];

    if (typeof localizedText === "string" && localizedText !== "") {
      node.innerHTML = localizedText.replace(/\n/g, "<br>");
    }
  });

  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    window.MathJax.typesetPromise();
  }
};

document.querySelectorAll("[data-language-toggle]").forEach((toggleRoot) => {
  const buttons = toggleRoot.querySelectorAll("[data-language-button]");

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      applyLanguage(button.dataset.languageButton || "ko");
    });
  });
});

const initialLanguage = getSavedLanguage();
applyLanguage(initialLanguage === "en" ? "en" : "ko");

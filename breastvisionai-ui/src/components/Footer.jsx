const LINKS = [
  // "Privacy Policy",
  // "Terms of Service",
  // "Documentation",
  // "Support",
];

export default function Footer({ simple = false }) {
  if (simple) {
    return (
      <footer className="border-t border-outline-variant pt-6 pb-2 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="font-headline-sm text-headline-sm text-on-surface flex items-center gap-2">
          <img
            src="/logo.png"
            alt="BreastVisionAI"
            width={40}
            height={40}
            className="material-symbols-outlined"
            style={{ fontSize: "20px" }}
          ></img>
          BreastVisionAI
        </div>
        <div className="font-body-sm text-body-sm text-on-surface-variant text-center sm:text-left">
          © 2026 BreastVisionAI Clinical Research Platform. All medical data
          encrypted.
        </div>
        <nav className="flex gap-4 font-body-sm text-body-sm text-on-surface-variant">
          {LINKS.map((label) => (
            <a
              key={label}
              href="#"
              className="hover:text-primary transition-colors duration-200"
            >
              {label}
            </a>
          ))}
        </nav>
      </footer>
    );
  }

  return (
    <footer className="bg-surface-container-low w-full py-8 px-4 md:px-10 flex flex-col md:flex-row justify-between items-center gap-4 border-t border-outline-variant">
      <div className="font-headline-sm text-headline-sm text-on-surface flex items-center gap-2">
        <img
          src="/logo.png"
          alt="BreastVisionAI"
          width={40}
          height={40}
          className="material-symbols-outlined"
          style={{ fontSize: "20px" }}
        ></img>
        BreastVisionAI
      </div>
      <p className="font-body-sm text-body-sm text-secondary text-center sm:text-left">
        © 2026 BreastVisionAI Clinical Research Platform. All medical data
        encrypted.
      </p>
      <div className="flex gap-4">
        {LINKS.map((label) => (
          <a
            key={label}
            href="#"
            className="font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors duration-200"
          >
            {label}
          </a>
        ))}
      </div>
    </footer>
  );
}

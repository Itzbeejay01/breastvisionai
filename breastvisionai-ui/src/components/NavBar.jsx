import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import useStore from "../store/index";

const links = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/diagnosis", label: "Diagnosis" },
  { to: "/batch", label: "Batch" },
  { to: "/models", label: "Models" },
];

export default function NavBar() {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const user = useStore((state) => state.user);
  const logout = useStore((state) => state.logout);

  // Helper to toggle mobile state
  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <header className="app-nav bg-surface/95 backdrop-blur-md fixed top-0 left-0 w-full z-50 border-b border-outline-variant/60 h-20">
      <div className="max-w-container-max mx-auto px-5 md:px-10 h-full flex justify-between items-center relative bg-transparent">
        
        {/* Brand Logo & Name */}
        <div className="flex items-center gap-2 font-headline-md text-headline-md font-bold text-primary">
          <img 
            src="/logo.png" 
            alt="BreastVisionAI" 
            width={40} 
            height={40} 
          />
          <span>BreastVisionAI</span>
        </div>

        {/* Desktop Navigation Links */}
        <nav className="hidden md:flex gap-6 items-center h-full">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `font-body-md text-body-md transition-colors duration-200 py-1 border-b-2 ${
                  isActive
                    ? "text-primary font-bold border-primary"
                    : "text-on-surface-variant border-transparent hover:text-primary"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden md:flex items-center gap-3 pl-5 border-l border-outline-variant/50">
          <div className="w-9 h-9 rounded-full bg-primary-container text-primary grid place-items-center font-semibold text-sm">{user?.username?.slice(0, 1).toUpperCase() || "U"}</div>
          <button type="button" onClick={async () => { await logout(); navigate("/login", { replace: true }); }} className="text-sm font-semibold text-on-surface-variant hover:text-primary">Sign out</button>
        </div>

        {/* Mobile Toggle Button */}
        <div className="flex md:hidden items-center">
          <button
            type="button"
            onClick={toggleMenu}
            className="text-primary hover:text-primary-container transition-colors touch-target flex items-center justify-center p-2"
            aria-expanded={isOpen}
            aria-label="Toggle navigation menu"
          >
            <span className="material-symbols-outlined text-[32px]">
              {isOpen ? "close" : "menu"}
            </span>
          </button>
        </div>

      </div>

      {/* Mobile Dropdown Menu */}
      <div
        className={`absolute top-20 left-0 w-full bg-surface border-b border-outline-variant transition-all duration-300 md:hidden z-40 shadow-lg ${
          isOpen ? "opacity-100 visible translate-y-0" : "opacity-0 invisible -translate-y-2"
        }`}
      >
        <nav className="flex flex-col py-4 px-6 gap-4">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              onClick={closeMenu}
              className={({ isActive }) =>
                `font-body-lg text-body-lg py-2 transition-colors duration-200 ${
                  isActive
                    ? "text-primary font-bold"
                    : "text-on-surface-variant hover:text-primary"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

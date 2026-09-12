import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import useStore from "../store/index";

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const login = useStore((state) => state.login);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault(); setLoading(true); setError("");
    try { await login(username, password); navigate(location.state?.from?.pathname || "/dashboard", { replace: true }); }
    catch (err) { setError(err.response?.data?.error || "Unable to sign in. Check your credentials."); }
    finally { setLoading(false); }
  };

  return (
    <main className="login-page min-h-screen flex items-center justify-center p-5 sm:p-8">
      <div className="login-shell w-full max-w-[1080px] grid lg:grid-cols-[1.02fr_.98fr] overflow-hidden">
        <section className="login-visual hidden lg:flex flex-col justify-between p-12 xl:p-16 relative overflow-hidden text-white"><div className="login-grid-pattern" /><div className="login-orbit login-orbit-one" /><div className="login-orbit login-orbit-two" /><Link to="/" className="relative z-10 flex items-center gap-3 font-headline-md font-semibold"><span className="login-brand-mark"><img src="/logo.png" alt="" width="30" height="30" /></span>BreastVisionAI</Link><div className="relative z-10 max-w-lg"><div className="login-mini-label"><span /> SECURE RESEARCH WORKSPACE</div><h1 className="font-display text-6xl xl:text-7xl font-semibold leading-[.96] tracking-[-.06em] mt-6">Your next<br /><span>clearer</span><br />decision.</h1><p className="text-white/70 text-lg leading-8 mt-7 max-w-md">A focused clinical AI workspace for breast imaging, ensemble analysis, and explainable results.</p></div><div className="relative z-10 flex items-center gap-3 text-sm text-white/65"><span className="material-symbols-outlined text-[#a9fff1]">shield_lock</span> Protected session · Research use only</div></section>
        <section className="login-form-panel p-7 sm:p-12 xl:p-16"><Link to="/" className="login-mobile-brand lg:hidden"><span className="login-brand-mark"><img src="/logo.png" alt="" width="27" height="27" /></span>BreastVisionAI</Link><div className="login-form-content"><div className="login-welcome-icon"><span className="material-symbols-outlined">fingerprint</span></div><p className="dashboard-kicker mt-7">WELCOME BACK</p><h2 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mt-3">Sign in to<br /><span className="login-gradient-text">your workspace.</span></h2><p className="text-on-surface-variant leading-7 mt-4">Authenticate before uploading or processing clinical images.</p><form onSubmit={submit} className="mt-9 space-y-5"><label className="login-field"><span>Username</span><div><span className="material-symbols-outlined">person</span><input value={username} onChange={(event) => setUsername(event.target.value)} required autoComplete="username" placeholder="Enter your username" /></div></label><label className="login-field"><span>Password</span><div><span className="material-symbols-outlined">lock</span><input type="password" value={password} onChange={(event) => setPassword(event.target.value)} required autoComplete="current-password" placeholder="Enter your password" /></div></label>{error && <p className="login-error"><span className="material-symbols-outlined">error</span>{error}</p>}<button type="submit" disabled={loading} className="login-submit">{loading ? "Signing in…" : "Continue securely"}<span className="material-symbols-outlined">arrow_forward</span></button></form><div className="login-divider"><span /> <small>AUTHORIZED PERSONNEL ONLY</small> <span /></div><p className="text-center text-sm text-on-surface-variant"><Link to="/" className="text-primary font-semibold hover:underline">Return to overview</Link></p></div></section>
      </div>
    </main>
  );
}

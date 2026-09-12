import { Navigate, Outlet, useLocation } from "react-router-dom";
import useStore from "../store/index";

export default function ProtectedRoute() {
  const location = useLocation();
  const authStatus = useStore((state) => state.authStatus);
  if (authStatus === "checking") return <div className="min-h-screen grid place-items-center text-primary"><span className="material-symbols-outlined animate-spin text-3xl">progress_activity</span></div>;
  return authStatus === "authenticated" ? <Outlet /> : <Navigate to="/login" replace state={{ from: location }} />;
}

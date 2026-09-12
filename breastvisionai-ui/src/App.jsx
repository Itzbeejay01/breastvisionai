import { useEffect } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import NavBar from './components/NavBar.jsx'
import WelcomePage from './pages/WelcomePage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import DiagnosisPage from './pages/DiagnosisPage.jsx'
import BatchPage from './pages/BatchPage.jsx'
import ModelComparisonPage from './pages/ModelComparisonPage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import ProtectedRoute from './components/ProtectedRoute.jsx'
import useStore from './store/index'

function App() {
  const location = useLocation()
  const isPublic = location.pathname === '/' || location.pathname === '/login'
  const initializeAuth = useStore((state) => state.initializeAuth)

  useEffect(() => { initializeAuth() }, [initializeAuth])

  return (
    <div className="min-h-screen flex flex-col bg-surface text-on-surface font-sans">
      {!isPublic && <NavBar />}
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/diagnosis" element={<DiagnosisPage />} />
          <Route path="/diagnosis/:id" element={<DiagnosisPage />} />
          <Route path="/batch" element={<BatchPage />} />
          <Route path="/models" element={<ModelComparisonPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}

export default App

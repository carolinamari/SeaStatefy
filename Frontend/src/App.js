import { BrowserRouter, Routes, Route } from 'react-router-dom'

import './App.css';
import { HOME, TOOL, ABOUT } from './utils/constants'
import Header from './components/header/Header';
import HomePage from './components/homePage/HomePage';
import ToolPage from './components/toolPage/ToolPage';
import AboutPage from './components/aboutPage/AboutPage';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={
                    <>
                        <Header currentPage={HOME} />
                        <HomePage />
                    </>
                }/>
                <Route path="/tool" element={
                    <>
                        <Header currentPage={TOOL} />
                        <ToolPage />
                    </>
                }/>
                <Route path="/about" element={
                    <>
                        <Header currentPage={ABOUT} />
                        <AboutPage />
                    </>
                } />
            </Routes>
        </BrowserRouter>    
    );
}

export default App;

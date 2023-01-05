import React from 'react'
import { Link } from 'react-router-dom'

import './header.css'
import HeaderButton from '../headerButton/HeaderButton'
import { HOME, TOOL, ABOUT } from '../../utils/constants'
import wave_icon from '../../assets/wave_icon.png'

const Header = ({ currentPage }) => {
    return (
        <div className='header'>
            <div className='logo-wrapper'>
                <img className='logo-img' src={wave_icon} alt='Website logo'></img>
            </div>
            <div className='buttons-wrapper'>
                <Link to='/'><HeaderButton name={HOME} selected={currentPage === HOME} /></Link>
                <Link to='/tool'><HeaderButton name={TOOL} selected={currentPage === TOOL} /></Link>
                <Link to='/about'><HeaderButton name={ABOUT} selected={currentPage === ABOUT} /></Link>
            </div>
        </div>
    )
}

export default Header
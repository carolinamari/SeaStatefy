import React from 'react'
import './headerButton.css'

const HeaderButton = ({ name, selected }) => {
    return (
        <button className={`header-button ${selected ? 'selected' : 'not-selected'}`}>{name}</button>
    )
}

export default HeaderButton
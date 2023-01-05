import React from 'react'

import './defaultButton.css'

const DefaultButton = ({ name, onClick, style }) => {
    return (
        <button className='default-button' onClick={onClick} style={style}>{name}</button>
    )
}

export default DefaultButton
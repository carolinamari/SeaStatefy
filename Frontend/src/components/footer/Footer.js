import React from 'react'

import './footer.css'

const Footer = ({ iconAuthorsList, style }) => {
    const authors = iconAuthorsList.join(', ')

    return (
        <p className='credits-text' style={style}>Icons made by <b>{authors}</b> from <b>www.flaticon.com</b></p>
    )
}

export default Footer
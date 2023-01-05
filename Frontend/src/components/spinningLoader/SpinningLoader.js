import React from 'react'

import './spinningLoader.css'
import { ProgressSpinner } from 'primereact/progressspinner';

const SpinningLoader = () => {
    return (
        <ProgressSpinner className='custom-spinner' />
    )
}

export default SpinningLoader
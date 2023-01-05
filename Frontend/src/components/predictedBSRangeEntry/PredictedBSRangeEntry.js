import React from 'react'

import './predictedBSRangeEntry.css'

const PredictedBSRangeEntry = ({ label, text, lastLine }) => {
    const formattedText = 
        <>
            {lastLine ? 
            <>
                <div>Entre</div>
                <ul className='bullet-style'>
                    <li>{`${text.from} e;`}</li>
                    <li>{text.to}</li>
                </ul>
            </> : 
            <span>{text}</span>}
        </>

    return (
    <>
        <div className='entry-wrapper'>
            <div className={`entry-label ${lastLine ? 'last-label' : ''}`}>{label}</div>
            <div className='entry-text'>{formattedText}</div>
        </div>
        {lastLine ? <></> : <div className='bottom-line'></div>}
    </>
  )
}

export default PredictedBSRangeEntry
import React from 'react'

import './tutorialStepBlock.css'

const TutorialStepBlock = ({ textFirst, stepNumber, stepText, stepImage }) => {
    const text = <>
        <div className='step-description'>
            <div className='step-number'>{stepNumber}</div>
            <p className='step-text'>{stepText}</p>
        </div>
    </>

    const img = <>
        <div className='step-img-wrapper'>
            <img className='step-img' src={stepImage} alt=''></img>
        </div>
    </>

    return (
        <div className='step-wrapper'>
            {textFirst ? <>{text} {img}</> : <>{img} {text}</> }
        </div>
    )
}

export default TutorialStepBlock
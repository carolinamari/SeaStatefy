import React from 'react'

import './homePage.css'
import { TITLE, DESCRIPTION, START_BUTTON_NAME, TUTORIAL_TITLE, TUTORIAL_STEP_1, TUTORIAL_STEP_2, TUTORIAL_STEP_3, TUTORIAL_STEP_4 } from '../../utils/constants'
import home_tool from '../../assets/home_tool.png'
import home_step1 from '../../assets/home_step1.png'
import home_step2 from '../../assets/home_step2.png'
import home_step3 from '../../assets/home_step3.png'
import home_step4 from '../../assets/home_step4.png'
import DefaultButton from '../defaultButton/DefaultButton'
import TutorialStepBlock from '../tutorialStepBlock/TutorialStepBlock'
import Footer from '../footer/Footer'

const HomePage = () => {

    const onClickHandler = (e) => {
        e.preventDefault();
        window.location.href='/tool';
    }

    return (
        <>
            <div className='intro-section'>
                <div className='intro-text'>
                    <div>
                        <h1 className='home-title'>{TITLE}</h1>
                        <p className='home-description'>{DESCRIPTION}</p>
                    </div>
                    <DefaultButton name={START_BUTTON_NAME} onClick={onClickHandler} />
                </div>
                <div className='intro-img-wrapper'>
                    <img className='intro-img' src={home_tool} alt=''></img>
                </div>
            </div>
            <div className='tutorial-section'>
                <h2 className='tutorial-subtitle'>{TUTORIAL_TITLE}</h2>
                <TutorialStepBlock textFirst={false} stepNumber='1' stepText={TUTORIAL_STEP_1} stepImage={home_step1} />
                <TutorialStepBlock textFirst={true} stepNumber='2' stepText={TUTORIAL_STEP_2} stepImage={home_step2} />
                <TutorialStepBlock textFirst={false} stepNumber='3' stepText={TUTORIAL_STEP_3} stepImage={home_step3} />
                <TutorialStepBlock textFirst={true} stepNumber='4' stepText={TUTORIAL_STEP_4} stepImage={home_step4} />
            </div>
            <Footer iconAuthorsList={['Freepik' ,'Icongeek']} style={{background: '#FAF9F6'}} />
        </>
    )
}

export default HomePage
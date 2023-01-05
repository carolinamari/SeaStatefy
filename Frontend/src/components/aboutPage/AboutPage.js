import React from 'react'

import './aboutPage.css'
import info_motivation_img from '../../assets/info_motivation_img.png'
import info_objective_img from '../../assets/info_objective_img.png'
import info_metodology_img from '../../assets/info_metodology_img.png'
import accuracy_table from '../../assets/accuracy_table.png'
import confusion_matrix from '../../assets/confusion_matrix.png'
import minerva_logo from '../../assets/minerva_logo.png'
import { PROJECT_NAME, PROJECT_SUMMARY, MOTIVATION_SECTION_NAME, MOTIVATION_TEXT, MOTIVATION_TEXT_REFS, OBJECTIVE_SECTION_NAME, OBJECTIVE_TEXT, OBJECTIVE_ITEMS, METODOLOGY_SECTION_NAME, METODOLOGY_TEXT, RESULTS_SECTION_NAME, RESULTS_TEXT, TCC_INFO } from '../../utils/constants'
import Footer from '../footer/Footer'

const AboutPage = () => {
    return (
        <div className='about-page'>
            <h1 className='project-name'>{PROJECT_NAME}</h1>
            <p className='project-summary'>{PROJECT_SUMMARY}</p>
            <div className='section-wrapper section-motivation'>
                <div className='text-motivation'>
                    <h2 className='section-title'>{MOTIVATION_SECTION_NAME}</h2>
                    { MOTIVATION_TEXT.split('\n').map(str => <p className='section-text'>{str}</p>) }
                    { MOTIVATION_TEXT_REFS.split('\n').map(str => <p className='section-ref'>{str}</p>) }
                </div>
                <div className='section-img-wrapper img-wrapper-motivation'>
                    <img className='section-img' src={info_motivation_img} alt=''></img>
                </div>
            </div>
            <div className='section-wrapper section-objective'>
                <div className='section-img-wrapper img-wrapper-objective'>
                    <img className='section-img' src={info_objective_img} alt=''></img>
                </div>
                <div className='text-objective'>
                    <h2 className='section-title'>{OBJECTIVE_SECTION_NAME}</h2>
                    { OBJECTIVE_TEXT.split('\n').map(str => <p className='section-text'>{str}</p>) }
                    <ul style={{marginTop: 0}}>
                        {OBJECTIVE_ITEMS.split('\n').map(i => <li className='section-text'>{i}</li>)}
                    </ul>
                </div>
            </div>
            <div className='section-wrapper section-metodology'>
                <div className='text-metodology'>
                    <h2 className='section-title'>{METODOLOGY_SECTION_NAME}</h2>
                    { METODOLOGY_TEXT.split('\n').map(str => <p className='section-text'>{str}</p>) }
                </div>
                <div className='section-img-wrapper img-wrapper-metodology'>
                    <img className='section-img' src={info_metodology_img} alt=''></img>
                </div>
            </div>
            <div className='section-wrapper section-results'>
                <div className='text-results'>
                    <h2 className='section-title title-results'>{RESULTS_SECTION_NAME}</h2>
                    { RESULTS_TEXT.split('\n').map(str => <p className='section-text'>{str}</p>) }
                </div>
                <div className='section-img-wrapper img-wrapper-results-table'>
                    <img className='section-img' src={accuracy_table} alt=''></img>
                </div>
                <div className='section-img-wrapper img-wrapper-results-matrix'>
                    <img className='section-img' src={confusion_matrix} alt=''></img>
                </div>
            </div>
            <div className='section-wrapper section-tcc'>
                <div className='section-img-wrapper img-wrapper-tcc'>
                    <img className='section-img' src={minerva_logo} alt=''></img>
                </div>
                <div className='text-tcc'>
                    { TCC_INFO.split('\n').map(str => <p className='section-text'>{str}</p>) }
                    <ul style={{marginTop: 0}}>
                        <li className='section-text'><a target="_blank" rel="noopener noreferrer" href='https://github.com/carolinamari/SeaStatefy'>Github</a></li>
                        <li className='section-text'><a target="_blank" rel="noopener noreferrer" href='https://youtu.be/PutPWx9PhrE'>Demo</a></li>
                        <li className='section-text'><a target="_blank" rel="noopener noreferrer" href='https://pcs.usp.br/pcspf/wp-content/uploads/sites/8/2022/12/Monografia_PCS3560_SEM_2022_Grupo_S09.pdf'>Monografia</a></li>
                    </ul>
                </div>
            </div>
            <Footer iconAuthorsList={['Freepik', 'Icongeek', 'Eucalyp']} style={{background: '#F5EEE5'}} />
        </div>
    )
}

export default AboutPage
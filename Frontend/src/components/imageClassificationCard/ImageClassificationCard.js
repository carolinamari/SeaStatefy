import React from 'react'
import DefaultButton from '../defaultButton/DefaultButton'

import './imageClassificationCard.css'
import { NEW_IMAGE_BUTTON, PREDICTED_BS_INTERVAL_TITLE, PREDICTED_PROBABILITIES_TITLE, SIMILAR_IMAGES_TITLE, SIMILAR_IMAGES_INFO } from '../../utils/constants'
import PredictedBSRangeTable from '../predictedBSRangeTable/PredictedBSRangeTable'
import { ProbabilityChart } from '../probabilityChart/ProbabilityChart'
import { Tooltip } from 'primereact/tooltip';
import selectSampleImage from '../../utils/selectSampleImage'


const ImageClassificationCard = ({ setHasUploaded, imageURL, classId, classRange, description, windSpeed, meanWaveHeight, seaAppearance, probabilities }) => {

    const buttonStyle = {
        fontWeight: 500,
        fontSize: '1.2vw',
        margin: '8vh 0 3vh 6.5vw',
        alignSelf: 'flex-start'
    }

    const loadNewImageHandler = () => {
        setHasUploaded(false)
    }

    return (
        <div className='results-card-wrapper'>
            <div className='results-card'>
                <div className='preview-column'>
                    <img alt='Imagem do mar classificada' role="presentation" src={imageURL} className='img-preview' />
                    <DefaultButton name={NEW_IMAGE_BUTTON} style={buttonStyle} onClick={loadNewImageHandler} />
                </div>
                <div className='results-container'>
                    <div className='beaufort-scale-section'>
                        <h2 className='results-section-title'>{PREDICTED_BS_INTERVAL_TITLE}</h2>
                        <PredictedBSRangeTable classRange={classRange} description={description} windSpeed={windSpeed} meanWaveHeight={meanWaveHeight} seaAppearance={seaAppearance} />
                    </div>
                    <div className='section-bottom-line'></div>
                    <div className='additional-info'>
                        <div className='probabilities-section'>
                            <h2 className='results-section-title'>{PREDICTED_PROBABILITIES_TITLE}</h2>
                            <div style={{ paddingLeft:'5.5vw' }}>
                                <ProbabilityChart classId={classId} probabilities={probabilities} />
                            </div>
                        </div>
                        <div className='section-lateral-line'></div>
                        <div className='similar-imgs-section'>
                            <div className='imgs-section-header'>
                                <h2 className='results-section-title'>{SIMILAR_IMAGES_TITLE}</h2>
                                <div className="icon-tooltip-wrapper">
                                    <Tooltip target=".info-icon" />
                                    <i className='pi pi-question-circle info-icon' data-pr-tooltip={SIMILAR_IMAGES_INFO} data-pr-position="bottom"></i>
                                </div>
                            </div>
                            <div style={{ paddingLeft: '3vw' }}>
                                <img alt='Imagem do mar exemplo 1' role="presentation" src={selectSampleImage({ classId:classId, imgId: 1 })} className='sample-thumbnail' />
                                <img alt='Imagem do mar exemplo 2' role="presentation" src={selectSampleImage({ classId:classId, imgId: 2 })} className='sample-thumbnail' />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ImageClassificationCard
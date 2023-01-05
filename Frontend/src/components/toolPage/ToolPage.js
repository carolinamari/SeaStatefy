import React from 'react'
import { useState } from 'react'

import './toolPage.css'
import ImageUploadCard from '../imageUploadCard/ImageUploadCard';
import ImageClassificationCard from '../imageClassificationCard/ImageClassificationCard';
import Footer from '../footer/Footer'
 

const ToolPage = () => {
    const [hasUploaded, setHasUploaded] = useState(false)
    const [imageURL, setImageURL] = useState(null)

    const [classId, setClassId] = useState()
    const [classRange, setClassRange] = useState()
    const [description, setDescription] = useState()
    const [windSpeed, setWindSpeed] = useState()
    const [meanWaveHeight, setMeanWaveHeight] = useState()
    const [seaAppearance, setSeaAppearance] = useState()
    const [probabilities, setProbabilities] = useState()

    return (
        <>
            <div className='tool-page'>
                {!hasUploaded ? 
                    <ImageUploadCard setHasUploaded={setHasUploaded} setImageURL={setImageURL} setClassId={setClassId} setClassRange={setClassRange} setDescription={setDescription} setWindSpeed={setWindSpeed} setMeanWaveHeight={setMeanWaveHeight} setSeaAppearance={setSeaAppearance} setProbabilities={setProbabilities} /> : 
                    <ImageClassificationCard setHasUploaded={setHasUploaded} imageURL={imageURL} classId={classId} classRange={classRange} description={description} windSpeed={windSpeed} meanWaveHeight={meanWaveHeight} seaAppearance={seaAppearance} probabilities={probabilities} />
                }
            </div>
            <Footer iconAuthorsList={['Icongeek']} style={{background: '#FAF9F6', paddingTop: '3vh'}} />
        </>
    )
}

export default ToolPage
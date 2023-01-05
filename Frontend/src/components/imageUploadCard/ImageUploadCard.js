import React from 'react'
import { useState, useRef } from 'react'

import 'primeicons/primeicons.css';
import './imageUploadCard.css'
import { UPLOAD_IMAGE_TITLE, SUPPORTED_FORMATS, DRAG_AND_DROP_TEXT, UPLOAD_IMAGE_BUTTON, MAX_FILE_NAME_LENGTH } from '../../utils/constants'
import { BASE_URL, CLASSIFY_ENDPOINT, RESPONSE_WAVE_HEIGHT_LABEL, RESPONSE_SEA_APPEARANCE_LABEL, RESPONSE_CLASS_LABEL, RESPONSE_CLASS_ID_LABEL, RESPONSE_DESCRIPTION_LABEL, RESPONSE_PROBABILITIES_LABEL, RESPONSE_WIND_SPEED_LABEL } from '../../utils/api'
import { FileUpload } from 'primereact/fileupload';
import DefaultButton from '../defaultButton/DefaultButton';
import validateSelectedImage from '../../utils/validateSelectedImage'
import ImageUploadErrorModal from '../imageUploadErrorModal/ImageUploadErrorModal';
import ApiResponseErrorModal from '../apiResponseErrorModal/ApiResponseErrorModal';
import SpinningLoader from '../spinningLoader/SpinningLoader';


const ImageUploadCard = ({ setHasUploaded, setImageURL, setClassId, setClassRange, setDescription, setWindSpeed, setMeanWaveHeight, setSeaAppearance, setProbabilities }) => {
    
    const [hasSelected, setHasSelected] = useState(false)
    const [error, setError] = useState(false)
    const [apiError, setApiError] = useState(false)
    const [isLoading, setIsLoading] = useState(false)

    const fileUploadRef = useRef(null);

    const chooseOptions = { icon: 'pi pi-image', iconOnly: true, className: 'custom-button custom-choose-button' };
    const cancelOptions = { icon: 'pi pi-times', iconOnly: true, className: 'custom-button custom-cancel-button' };
    
    const buttonStyle = {
        fontWeight: 500,
        fontSize: '1.2vw'
    }

    const headerTemplate = (options) => {
        const { chooseButton, cancelButton } = options;

        return (
            <div className='custom-buttons-container' style={ hasSelected ? { visibility: 'visible' } : { visibility: 'hidden' }}>
                {chooseButton}
                {cancelButton}
            </div>
        );
    };

    const emptyTemplate = (props) => {
        return (
            <div className='empty-content-wrapper'>
                <div className='empty-content'>
                    <p className='empty-content-text'>{DRAG_AND_DROP_TEXT}</p>
                    <div>
                        <i className="pi pi-image empty-content-icon"></i>
                    </div>
                    <DefaultButton name={UPLOAD_IMAGE_BUTTON} onClick={() => fileUploadRef.current.getInput().click()} style={buttonStyle} />
                </div>
            </div>
        )
    }

    const itemTemplate = (file, props) => {
        return (
            <div className='item-content-wrapper'>
                <div className="item-content-summary">
                    <img alt={file.name} role="presentation" src={file.objectURL} className='item-preview' />
                    <div className="item-info">
                        <div className='item-name'>{file.name.length > MAX_FILE_NAME_LENGTH ? file.name.substring(0, MAX_FILE_NAME_LENGTH) + '...' : file.name}</div>
                        <div className='item-size'>{props.formatSize}</div>
                        <DefaultButton name='Classificar' onClick={uploadButtonHandler} style={buttonStyle} />
                    </div>
                    
                </div>
            </div>
        );
    };

    const onTemplateSelect = (e) => {
        const file = e.files[0]
        if (validateSelectedImage(file)) {
            setImageURL(file.objectURL)
            setHasSelected(true)
        } else {
            fileUploadRef.current.clear()
            setError(true)
        }
    }

    const onTemplateRemove = (e) => {
        setHasSelected(false)
    }

    const onTemplateClear = (e) => {
        setHasSelected(false)
    }

    const uploadButtonHandler = async (e) => {
        setIsLoading(true)
        fileUploadRef.current.upload()
    }

    const uploadHandler = async (e) => {
        // Convert image to base64 encoded
        const file = e.files[0]
        const reader = new FileReader()
        let blob = await fetch(file.objectURL).then((r) => r.blob()) //blob:url

        reader.readAsDataURL(blob)

        reader.onloadend = () => {
            const base64data = reader.result.split(',')[1]
            fetch(BASE_URL + CLASSIFY_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    img: base64data
                })
            }).then((response) => {
                console.log(response)
                if (response.status === 200) {
                    return response.json()
                } else {
                    setApiError(true)
                    return
                }
            })
            .catch(error => {
                console.log(error)
                setIsLoading(false)
                setApiError(true)
            })
            .then(data => {
                setIsLoading(false)
                if (data) {
                    console.log(data)
                    setClassId(data[RESPONSE_CLASS_ID_LABEL])
                    setClassRange(data[RESPONSE_CLASS_LABEL])
                    setDescription(data[RESPONSE_DESCRIPTION_LABEL])
                    setWindSpeed(data[RESPONSE_WIND_SPEED_LABEL])
                    setMeanWaveHeight(data[RESPONSE_WAVE_HEIGHT_LABEL])
                    setSeaAppearance(data[RESPONSE_SEA_APPEARANCE_LABEL])
                    setProbabilities(data[RESPONSE_PROBABILITIES_LABEL])
                    setHasUploaded(true)
                }
            })
        }
    }

    return (
        <>
            {isLoading ? <SpinningLoader/> : <></>}
            <div className='upload-card'>
                <h1 className='upload-card-heading'>{UPLOAD_IMAGE_TITLE}</h1>
                <p className='upload-card-formats'>{SUPPORTED_FORMATS}</p>
                <FileUpload
                    ref={fileUploadRef}
                    name="img[]"
                    url={BASE_URL + CLASSIFY_ENDPOINT}
                    accept="image/*"
                    customUpload
                    uploadHandler={uploadHandler}
                    onSelect={onTemplateSelect}
                    onClear={onTemplateClear}
                    onRemove={onTemplateRemove}
                    headerTemplate={headerTemplate}
                    itemTemplate={itemTemplate}
                    emptyTemplate={emptyTemplate}
                    chooseOptions={chooseOptions}
                    cancelOptions={cancelOptions}
                />
                <ImageUploadErrorModal error={error} onHide={() => setError(false)} />
                <ApiResponseErrorModal apiError={apiError} onHide={() => setApiError(false)} />
            </div>
        </>
    )
}

export default ImageUploadCard
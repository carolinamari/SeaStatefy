import React from 'react'

import './imageUploadErrorModal.css'
import { Dialog } from 'primereact/dialog';
import { FILE_UPLOAD_ERROR_TITLE, FILE_UPLOAD_ERROR_MESSAGE } from '../../utils/constants'

const ImageUploadErrorModal = ({ error, onHide }) => {
    const renderHeader = () => {
        return (
            <div className='modal-header'>
                <i className='pi pi-exclamation-triangle modal-header-icon'></i>
                <div className='modal-header-title'>{FILE_UPLOAD_ERROR_TITLE}</div>
                <i className='pi pi-exclamation-triangle modal-header-icon' style={{visibility: 'hidden'}}></i>
            </div>
        )
    }
    
    const renderFooter = () => {
        return (
            <div className='modal-footer'>
                <button onClick={() => onHide()} className="dismiss-footer-button">OK</button>
            </div>
        )
    }
    
    return (
        <Dialog header={renderHeader()} visible={error} className='modal-container' contentClassName='modal-content' footer={renderFooter()} onHide={onHide} dismissableMask={true}>
            <p className='modal-message'>{FILE_UPLOAD_ERROR_MESSAGE}</p>
        </Dialog>
    )
}

export default ImageUploadErrorModal
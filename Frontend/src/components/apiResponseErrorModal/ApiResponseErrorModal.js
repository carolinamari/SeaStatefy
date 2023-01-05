import React from 'react'

import './apiResponseErrorModal.css'
import { Dialog } from 'primereact/dialog';
import { API_RESPONSE_ERROR_TITLE, API_RESPONSE_ERROR_MESSAGE } from '../../utils/constants'

const ApiResponseErrorModal = ({ apiError, onHide }) => {
    const renderHeader = () => {
        return (
            <div className='modal-header'>
                <i className='pi pi-exclamation-triangle modal-header-icon'></i>
                <div className='modal-header-title'>{API_RESPONSE_ERROR_TITLE}</div>
                <i className='pi pi-exclamation-triangle modal-header-icon' style={{visibility: 'hidden'}}></i>
            </div>
        )
    }
    
    const renderFooter = () => {
        return (
            <div className='modal-footer'>
                <button onClick={() => onHide()} className="dismiss-footer-button">Fechar</button>
            </div>
        )
    }
    
    return (
        <Dialog header={renderHeader()} visible={apiError} className='modal-container' contentClassName='modal-content' footer={renderFooter()} onHide={onHide} dismissableMask={true}>
            <p className='modal-message'>{API_RESPONSE_ERROR_MESSAGE}</p>
        </Dialog>
    )
}

export default ApiResponseErrorModal
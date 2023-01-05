import React from 'react'
import PredictedBSRangeEntry from '../predictedBSRangeEntry/PredictedBSRangeEntry'

import './predictedBSRangeTable.css'
import { CLASS_LABEL, DESCRIPTION_LABEL, WIND_SPEED_LABEL, WAVE_HEIGHT_LABEL, SEA_APPEARANCE_LABEL } from '../../utils/constants'

const PredictedBSRangeTable = ({ classRange, description, windSpeed, meanWaveHeight, seaAppearance }) => {
    return (
        <div className='table-wrapper'>
            <PredictedBSRangeEntry label={CLASS_LABEL} text={classRange} lastLine={false}/>
            <PredictedBSRangeEntry label={DESCRIPTION_LABEL} text={description} lastLine={false} />
            <PredictedBSRangeEntry label={WIND_SPEED_LABEL} text={windSpeed} lastLine={false} />
            <PredictedBSRangeEntry label={WAVE_HEIGHT_LABEL} text={meanWaveHeight} lastLine={false} />
            <PredictedBSRangeEntry label={SEA_APPEARANCE_LABEL} text={seaAppearance} lastLine={true} />
        </div>
  )
}

export default PredictedBSRangeTable
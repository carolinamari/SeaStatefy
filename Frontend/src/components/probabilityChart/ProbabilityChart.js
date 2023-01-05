import React from 'react'

import './probabilityChart.css'
import {
    BarChart,
    Bar,
    Cell,
    LabelList,
    XAxis,
    YAxis,
    CartesianGrid,
    ResponsiveContainer,
  } from 'recharts';

export const ProbabilityChart = ({ classId, probabilities }) => {

    const barLabelFormatter = (value) => {
        return Math.round(value * 100.0) / 100.0 + '%'
    }

    return (
        <div style={{height: '32vh'}}>
            <ResponsiveContainer width="85%" height='100%'>
                <BarChart
                width={500}
                height={300}
                data={probabilities}
                layout='vertical'
                margin={{ top: 5, right: 20, left: 10, bottom: 5, }}
                >
                    <CartesianGrid horizontal={false} strokeDasharray='5 5' />
                    <XAxis 
                    type='number'
                    axisLine={false} 
                    tickLine={false}
                    tick={{fontSize: '0.9vw'}}
                    />
                    <YAxis
                    type='category'
                    dataKey='range' 
                    dx={-10}
                    axisLine={false} 
                    tickLine={false}
                    tick={{fontSize: '1vw', fill: '#23242A' }}
                    tickFormatter={(value) => value.toLocaleString().replace(/ /g, '\u00A0')}
                    />
                    <Bar 
                    dataKey='p' 
                    maxBarSize={30} 
                    radius={15} 
                    background={false}
                    label={barLabelFormatter}
                    >
                        <LabelList dataKey="p" position="right" formatter={(value) =>  Math.round(value * 100.0) / 100.0 + '%'} style={{ fontSize: '1vw' }}/>
                        { probabilities.map((entry, index) => index === classId - 1 ?
                        <Cell key={`cell-${index}`} fill='#24AEBB' /> :
                        <Cell key={`cell-${index}`} fill='#FFFFFF' />)
                        }
                    </Bar>    
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}

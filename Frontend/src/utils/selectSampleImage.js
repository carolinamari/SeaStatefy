import sample1_1 from '../assets/sample_images/sample1_1.JPG'
import sample1_2 from '../assets/sample_images/sample1_2.png'
import sample2_1 from '../assets/sample_images/sample2_1.JPG'
import sample2_2 from '../assets/sample_images/sample2_2.png'
import sample3_1 from '../assets/sample_images/sample3_1.png'
import sample3_2 from '../assets/sample_images/sample3_2.png'


const sample_img_dict = {
    'sample1_1': sample1_1,
    'sample1_2': sample1_2,
    'sample2_1': sample2_1,
    'sample2_2': sample2_2,
    'sample3_1': sample3_1,
    'sample3_2': sample3_2
}

export function selectSampleImage(input) {
    const { classId, imgId } = input
    const filename =`sample${classId}_${imgId}`
    return sample_img_dict[filename]
}

export default selectSampleImage
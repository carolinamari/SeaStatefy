export function validateSelectedImage(input) {
    // Validate file format (allowed types: .png, .jpeg and .tif)
    if (input.type ===  "image/png" || input.type === "image/jpeg" || input.type === "image/tiff") {
        return true
    } else {
        return false
    }
}



export default validateSelectedImage
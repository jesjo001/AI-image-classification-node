import fs, { appendFileSync } from 'fs'
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';

const imageCat = `${__dirname}/assets/cat.jpg`
const imageDog = `${__dirname}/assets/dog.jpg`

const image = fs.readFileSync(imageCat);
const decodedImage = tf.node.decodeImage(image, 3)

async function App() {
    const model = await mobilenet.load()
    const predictions = await model.classify(decodedImage)
    console.log(`Prediction: ${JSON.stringify(predictions, undefined, 2)}`);
}

App()
package nl.avisi.labs.deeplearning.transferlearning.handson.trainers;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.springframework.core.io.ClassPathResource;

import javax.imageio.ImageIO;
import javax.validation.constraints.NotNull;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class FeatureMapExtractor {
    // sets the DL4J model-download folder to a local folder instead of default user home
    static {
        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");
        DL4JResources.setBaseDirectory(new File("dl4j-models/"));
    }

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FeatureMapExtractor.class);

    private static final int NUM_CLASSES = 2; //Banana / No_banana
    private static final long SEED = 12345; //Should normally be random, but is set in this case to be get reproducible results

    private static final int TRAIN_PERC = 80; // Percentage of images that should be included in the trainings set, the rest is included in the test set
    private static final int BATCH_SIZE = 5;
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;

    private static int[] inputShape = new int[]{ 3, 224, 224 };
    private static int numClasses = 1000;
    private static IUpdater updater = new Nesterovs();
    private static CacheMode cacheMode = CacheMode.NONE;
    private static WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    private static ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private static Color bgColor = new Color(255, 255, 255);

    private static Color borderColor = new Color(140, 140, 140);

    private static NativeImageLoader imageLoader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    private static ComputationGraph computationGraph;

    private enum Orientation {
        LANDSCAPE, PORTRAIT
    }

    public static void main(String[] args) throws IOException {

        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(SEED)
                                                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                    .updater(updater)
                                                    .activation(Activation.RELU)
                                                    .cacheMode(cacheMode)
                                                    .trainingWorkspaceMode(workspaceMode)
                                                    .inferenceWorkspaceMode(workspaceMode)
                                                    .graphBuilder()
                                                    .addInputs("in")
                                                    // block 1
                                                    .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                                                                            .cudnnAlgoMode(cudnnAlgoMode).build(), "in")
                                                    .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode)
                                                                                            .build(), "0")
                                                    .layer(2, new SubsamplingLayer.Builder()
                                                            .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                            .stride(2, 2).build(), "1")
                                                    // block 2
                                                    .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode)
                                                                                            .build(), "2")
                                                    .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode)
                                                                                            .build(), "3")
                                                    .layer(5, new SubsamplingLayer.Builder()
                                                            .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                            .stride(2, 2).build(), "4")
                                                    // block 3
                                                    .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode)
                                                                                            .build(), "5")
                                                    .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode)
                                                                                            .build(), "6")
                                                    .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                            .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode)
                                                                                            .build(), "7")
                                                    .layer(9, new SubsamplingLayer.Builder()
                                                            .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                            .stride(2, 2).build(), "8")
                                                    // block 4
                                                    .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                             .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode)
                                                                                             .build(), "9")
                                                    .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                             .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode)
                                                                                             .build(), "10")
                                                    .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                                                             .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode)
                                                                                             .build(), "11")
                                                    .layer(13, new SubsamplingLayer.Builder()
                                                            .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                            .stride(2, 2).build(), "12")
//                                                    // block 5
//                                                    .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//                                                                                             .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode)
//                                                                                             .build(), "13")
//                                                    .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//                                                                                             .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode)
//                                                                                             .build(), "14")
//                                                    .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//                                                                                             .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode)
//                                                                                             .build(), "15")
//                                                    .layer(17, new SubsamplingLayer.Builder()
//                                                            .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
//                                                            .stride(2, 2).build(), "16")
                                                    //                .layer(18, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                                    //                        .build())
                                                    //                .layer(19, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                                    //                        .build())
//                                                    .layer(18, new OutputLayer.Builder(
//                                                            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
//                                                                                                             .nOut(numClasses).activation(Activation.SOFTMAX) // radial basis function required
//                                                                                                             .build(), "17")
                                                    .setOutputs("13")
//                                                    .backpropType(Back).pretrain(false)
                                                    .setInputTypes(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                                                    .build();
        ZooModel zooModel=  VGG16.builder().build();
        computationGraph = (ComputationGraph) zooModel.initPretrained();

//        computationGraph = new ComputationGraph(conf);
        computationGraph.init();
        Arrays.asList(computationGraph.getLayers()).forEach(layer -> layer.addListeners(new Listener(layer)));
        InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream("datasets/fruit/apple/apple2.jpg");
        BufferedImage sourceImage = ImageIO.read(is);
        INDArray image = loadImage(sourceImage);

        normalizeImage(image);
        BufferedImage bufferedImage = convert(Collections.singletonList(processImage(image)), sourceImage);

        File f = new File("saved.png");
        ImageIO.write(bufferedImage, "png", f);
    }

    private static class Listener extends BaseTrainingListener {

        private final Layer layer;

        public Listener(final org.deeplearning4j.nn.api.Layer layer) {
            this.layer = layer;
        }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {
            log.debug(model.toString());
        }

    }

    private static BufferedImage convert(List<INDArray> tensors3D, BufferedImage sourceImage) {

        long width = 0;
        long height = 0;

        int border = 1;
        int padding_row = 2;
        int padding_col = 80;

        boolean drawArrows = false;

        /*
            We determine height of joint output image. We assume that first position holds maximum dimensionality
         */
        long[] shape = tensors3D.get(0).shape();
        long numImages = shape[1];
        log.debug("Number of images {}", numImages);
        height = (shape[2]);
        width = (shape[3]);
        log.debug("Output image dimensions: {height: " + height + ", width: " + width + "}");
        int maxHeight = 0; //(height + (border * 2 ) + padding_row) * numImages;
        int totalWidth = 0;
        int iOffset = 1;
        Orientation orientation = Orientation.LANDSCAPE;
        List<BufferedImage> images = new ArrayList<>();
        for (int layer = 0; layer < tensors3D.size(); layer++) {
            INDArray tad = tensors3D.get(layer);
            int zoomed = 0;

            BufferedImage image = null;
            if (orientation == Orientation.LANDSCAPE) {
                maxHeight = (int) ((height + (border * 2) + padding_row) * numImages);
                image = renderMultipleImagesLandscape(tad, maxHeight, (int) width, (int) height, (int) numImages);
                totalWidth += image.getWidth() + padding_col;
            } else if (orientation == Orientation.PORTRAIT) {
                totalWidth = (int) ((width + (border * 2) + padding_row) * numImages);
                image = renderMultipleImagesPortrait(tad, totalWidth, (int) width, (int) height);
                maxHeight += image.getHeight() + padding_col;
            }

            images.add(image);
        }

        if (drawArrows) {
            if (orientation == Orientation.LANDSCAPE) {
                // append some space for arrows
                totalWidth += padding_col * 2;
            } else if (orientation == Orientation.PORTRAIT) {
                maxHeight += padding_col * 2;
                maxHeight += sourceImage.getHeight() + (padding_col * 2);
            }
        }


        BufferedImage output = new BufferedImage(totalWidth, maxHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = output.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, output.getWidth(), output.getHeight());

        BufferedImage singleArrow = null;
        BufferedImage multipleArrows = null;


        try {

            if (orientation == Orientation.LANDSCAPE) {
                try {
                    ClassPathResource resource = new ClassPathResource("arrow_sing.PNG");
                    ClassPathResource resource2 = new ClassPathResource("arrow_mul.PNG");

                    singleArrow = ImageIO.read(resource.getInputStream());
                    multipleArrows = ImageIO.read(resource2.getInputStream());
                } catch (Exception e) {
                }

                graphics2D.drawImage(sourceImage, (padding_col / 2) - (sourceImage.getWidth() / 2),
                        (maxHeight / 2) - (sourceImage.getHeight() / 2), null);

                graphics2D.setPaint(borderColor);
                graphics2D.drawRect((padding_col / 2) - (sourceImage.getWidth() / 2),
                        (maxHeight / 2) - (sourceImage.getHeight() / 2), sourceImage.getWidth(),
                        sourceImage.getHeight());

                iOffset += sourceImage.getWidth();

                if (drawArrows && singleArrow != null) {
                    graphics2D.drawImage(singleArrow, iOffset + (padding_col / 2) - (singleArrow.getWidth() / 2),
                            (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
                }
            } else {
                try {
                    ClassPathResource resource = new ClassPathResource("arrow_singi.PNG");
                    ClassPathResource resource2 = new ClassPathResource("arrow_muli.PNG");

                    singleArrow = ImageIO.read(resource.getInputStream());
                    multipleArrows = ImageIO.read(resource2.getInputStream());
                } catch (Exception e) {
                }

                graphics2D.drawImage(sourceImage, (totalWidth / 2) - (sourceImage.getWidth() / 2),
                        (padding_col / 2) - (sourceImage.getHeight() / 2), null);

                graphics2D.setPaint(borderColor);
                graphics2D.drawRect((totalWidth / 2) - (sourceImage.getWidth() / 2),
                        (padding_col / 2) - (sourceImage.getHeight() / 2), sourceImage.getWidth(),
                        sourceImage.getHeight());

                iOffset += sourceImage.getHeight();
                if (drawArrows && singleArrow != null) {
                    graphics2D.drawImage(singleArrow, (totalWidth / 2) - (singleArrow.getWidth() / 2),
                            iOffset + (padding_col / 2) - (singleArrow.getHeight() / 2), null);
                }

            }
            iOffset += padding_col;
        } catch (Exception e) {
            // if we can't load images - ignore them
        }



        /*
            now we merge all images into one big image with some offset
        */

        for (int i = 0; i < images.size(); i++) {
            BufferedImage curImage = images.get(i);
            if (orientation == Orientation.LANDSCAPE) {
                // image grows from left to right
                graphics2D.drawImage(curImage, iOffset, 1, null);
                iOffset += curImage.getWidth() + padding_col;

                if (singleArrow != null && multipleArrows != null) {
                    if (i < images.size() - 1) {
                        // draw multiple arrows here
                        if (multipleArrows != null) {
                            graphics2D.drawImage(multipleArrows,
                                    iOffset - (padding_col / 2) - (multipleArrows.getWidth() / 2),
                                    (maxHeight / 2) - (multipleArrows.getHeight() / 2), null);
                        }
                    } else {
                        // draw single arrow
                        //    graphics2D.drawImage(singleArrow, iOffset - (padding_col / 2) - (singleArrow.getWidth() / 2), (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
                    }
                }
            } else if (orientation == Orientation.PORTRAIT) {
                // image grows from top to bottom
                graphics2D.drawImage(curImage, 1, iOffset, null);
                iOffset += curImage.getHeight() + padding_col;

                if (singleArrow != null && multipleArrows != null) {
                    if (i < images.size() - 1) {
                        // draw multiple arrows here
                        if (multipleArrows != null) {
                            graphics2D.drawImage(multipleArrows, (totalWidth / 2) - (multipleArrows.getWidth() / 2),
                                    iOffset - (padding_col / 2) - (multipleArrows.getHeight() / 2), null);
                        }
                    } else {
                        // draw single arrow
                        //   graphics2D.drawImage(singleArrow, (totalWidth / 2) - (singleArrow.getWidth() / 2),  iOffset - (padding_col / 2) - (singleArrow.getHeight() / 2) , null);
                    }
                }
            }
        }

        return output;

    }

    private static BufferedImage renderMultipleImagesPortrait(INDArray tensor3D, int maxWidth, int zoomWidth, int zoomHeight) {
        int border = 1;
        int padding_row = 2;
        int padding_col = 2;
        int zoomPadding = 20;

        long[] tShape = tensor3D.shape();

        long numRows = tShape[0] / tShape[2];

        long height = (numRows * (tShape[1] + border + padding_col)) + padding_col + zoomPadding + zoomWidth;

        if (height > Integer.MAX_VALUE) {
            throw new ND4JArraySizeException();
        }
        BufferedImage outputImage = new BufferedImage(maxWidth, (int) height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = outputImage.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, outputImage.getWidth(), outputImage.getHeight());

        int columnOffset = 0;
        int rowOffset = 0;
        int numZoomed = 0;
        int limZoomed = 5;
        int zoomSpan = maxWidth / limZoomed;

        for (int z = 0; z < tensor3D.shape()[0]; z++) {

            INDArray tad2D = tensor3D.tensorAlongDimension(z, 2, 1);
            long[] shape = tad2D.shape();
            long rWidth = tad2D.shape()[0];
            long rHeight = tad2D.shape()[1];

            long loc_height = (rHeight) + (border * 2) + padding_row;
            long loc_width = (rWidth) + (border * 2) + padding_col;

            BufferedImage currentImage = renderImageGrayscale(tad2D);

            /*
                if resulting image doesn't fit into image, we should step to next columns
             */
            if (columnOffset + loc_width > maxWidth) {
                rowOffset += loc_height;
                columnOffset = 0;
            }

            /*
                now we should place this image into output image
            */

            graphics2D.drawImage(currentImage, columnOffset + 1, rowOffset + 1, null);


            /*
                draw borders around each image
            */

            graphics2D.setPaint(borderColor);
            graphics2D.drawRect(columnOffset, rowOffset, (int) tad2D.shape()[0], (int) tad2D.shape()[1]);



            /*
                draw one of 3 zoomed images if we're not on first level
            */

            if (z % 7 == 0 && // zoom each 5th element
                    z != 0 && // do not zoom 0 element
                    numZoomed < limZoomed && // we want only few zoomed samples
                    (rHeight != zoomHeight && rWidth != zoomWidth) // do not zoom if dimensions match
            ) {

                int cY = (zoomSpan * numZoomed) + (zoomHeight);
                int cX = (zoomSpan * numZoomed) + (zoomWidth);

                graphics2D.drawImage(currentImage, cX - 1, (int) height - zoomWidth - 1, zoomWidth, zoomHeight, null);
                graphics2D.drawRect(cX - 2, (int) height - zoomWidth - 2, zoomWidth, zoomHeight);

                // draw line to connect this zoomed pic with its original
                graphics2D.drawLine(columnOffset + (int) rWidth, rowOffset + (int) rHeight, cX - 2, (int) height - zoomWidth - 2);
                numZoomed++;

            }

            columnOffset += loc_width;
        }

        return outputImage;
    }

    private static BufferedImage renderMultipleImagesLandscape(INDArray tensor3D, int maxHeight, int zoomWidth,
                                                               int zoomHeight, int numImages) {
        /*
            first we need to determine size of the overall image
         */
        int border = 1;
        int padding_row = 2;
        int padding_col = 2;
        int zoomPadding = 20;

        long[] tShape = tensor3D.shape();

        long numColumns = tShape[0] / tShape[1];
        numColumns = 64;

        long width = (numColumns * (tShape[1] + border + padding_col)) + padding_col + zoomPadding + zoomWidth;

        BufferedImage outputImage = new BufferedImage((int) width, maxHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = outputImage.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, outputImage.getWidth(), outputImage.getHeight());

        int columnOffset = 0;
        int rowOffset = 0;
        int numZoomed = 0;
        int limZoomed = 5;
        int zoomSpan = maxHeight / limZoomed;
        System.out.println(tensor3D.shape()[0]);
        for (int z = 0; z < numImages /*64*/; z++) {

            INDArray tad2D = tensor3D.tensorAlongDimension(z, 3, 2);
            long[] tad2Dshape = tad2D.shape();
            long rWidth = tad2D.shape()[0];
            long rHeight = tad2D.shape()[1];

            long loc_height = (rHeight) + (border * 2) + padding_row;
            long loc_width = (rWidth) + (border * 2) + padding_col;

            BufferedImage currentImage = renderImageGrayscale(tad2D);

            /*
                if resulting image doesn't fit into image, we should step to next columns
             */
            if (rowOffset + loc_height > maxHeight) {
                columnOffset += loc_width;
                rowOffset = 0;
            }

            /*
                now we should place this image into output image
            */

            graphics2D.drawImage(currentImage, columnOffset + 1, rowOffset + 1, null);


            /*
                draw borders around each image
            */

            graphics2D.setPaint(borderColor);
            if (tad2D.shape()[0] > Integer.MAX_VALUE || tad2D.shape()[1] > Integer.MAX_VALUE) {
                throw new ND4JArraySizeException();
            }
            graphics2D.drawRect(columnOffset, rowOffset, (int) tad2D.shape()[0], (int) tad2D.shape()[1]);



            /*
                draw one of 3 zoomed images if we're not on first level
            */

            if (z % 5 == 0 && // zoom each 5th element
                    z != 0 && // do not zoom 0 element
                    numZoomed < limZoomed && // we want only few zoomed samples
                    (rHeight != zoomHeight && rWidth != zoomWidth) // do not zoom if dimensions match
            ) {

                int cY = (zoomSpan * numZoomed) + (zoomHeight);

                graphics2D.drawImage(currentImage, (int) width - zoomWidth - 1, cY - 1, zoomWidth, zoomHeight, null);
                graphics2D.drawRect((int) width - zoomWidth - 2, cY - 2, zoomWidth, zoomHeight);

                // draw line to connect this zoomed pic with its original
                graphics2D.drawLine(columnOffset + (int) rWidth, rowOffset + (int) rHeight, (int) width - zoomWidth - 2,
                        cY - 2 + zoomHeight);
                numZoomed++;
            }

            rowOffset += loc_height;
        }
        return outputImage;
    }

    /**
     * Renders 2D INDArray into BufferedImage
     *
     * @param array
     */
    private static BufferedImage renderImageGrayscale(INDArray array) {
        BufferedImage imageToRender = new BufferedImage(array.columns(), array.rows(), BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < array.columns(); x++) {
            for (int y = 0; y < array.rows(); y++) {
                imageToRender.getRaster().setSample(x, y, 0, (int) (255 * array.getRow(y).getDouble(x)));
            }
        }
        return imageToRender;
    }

    @NotNull
    private static NormalDistribution getDist() {
        return new NormalDistribution(0, 0.2 * (2.0 / (4096 + NUM_CLASSES)));
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()//
                                                  .updater(new Nesterovs(5e-5))//
                                                  .seed(SEED).build();
    }

    /**
     * Processes the image by feeding it through the network
     *
     * @param image
     * @return
     */
    private static INDArray processImage(final INDArray image) {
        INDArray[] output = computationGraph.output(false, image);
        return output[0];
    }

    private static INDArray loadImage(final InputStream inputStream) {
        INDArray image = null;
        try {
            image = imageLoader.asMatrix(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    private static INDArray loadImage(final BufferedImage bufferedImage) {
        INDArray image = null;
        try {
            image = imageLoader.asMatrix(bufferedImage);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    /**
     * Normalize the image
     *
     * @param image
     */
    private static void normalizeImage(final INDArray image) {
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
    }

}

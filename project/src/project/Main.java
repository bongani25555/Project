package project;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.*;

public class Main {
    public void run(String[] args) {
        // Declare the variables we are going to use
        Mat sr, src_gray = new Mat(), dst = new Mat();
        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        String window_name = "Laplace Demo";
        Mat src = Imgcodecs.imread("leaf.jpg",Imgcodecs.IMREAD_COLOR);
        String imageName = ((args.length > 0) ? args[0] : "C:/Users/User/Downloads/dataset/leaf.jpg");
        //src = Imgcodecs.imread(imageName, Imgcodecs.IMREAD_COLOR); // Load an image
        // Check if image is loaded fine
        if( src.empty() ) {
            System.out.println("Error opening image");
            System.out.println("Program Arguments: [image_name -- default ../data/lena.jpg] \n");
            System.exit(-1);
        }
        // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
        Imgproc.GaussianBlur( src, src, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT );
        // Convert the image to grayscale
        Imgproc.cvtColor( src, src_gray, Imgproc.COLOR_RGB2GRAY );
        Mat abs_dst = new Mat();
        Imgproc.Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, Core.BORDER_DEFAULT );
        Mat man= doCanny(dst);
        Imgcodecs.imwrite("C:/Users/User/Downloads/dataset/leaf.jpg", man);

        System.out.println("Image doCanny");

        man = doBackgroundRemoval(man);

        Imgcodecs.imwrite("C:/Users/User/Downloads/dataset/leaf2.jpg", man);
        COMatrix comatr=new COMatrix(man);
        int size1=comatr.NoOfgreylevels()+1;

        int[][] tempArray=new int[size1][size1];
        double[][] probOfIJ=new double[size1][size1];

        int angle=0;

        int d=1;
        int sum=0;


        for(int i=0;i<tempArray.length;i++) {
            for (int j = 0; j <tempArray[0].length; j++) {
                sum+=comatr.NoOfPairsOfPixelIJ(angle,d,i,j);
                tempArray[i][j]=comatr.NoOfPairsOfPixelIJ(angle,d,i,j);
            }
            System.out.println();
        }

        for (int i=0;i<tempArray.length;i++) {
            for (int j = 0; j < tempArray[0].length; j++) {
                probOfIJ[i][j] = (double) tempArray[i][j] / sum;
                System.out.print(tempArray[i][j] + " ");
            }
            System.out.println();
        }
        comatr.setupOfMean(probOfIJ,sum);
        comatr.setupOfVar(probOfIJ,sum);


        System.out.print('\n'+"Energy of P0,1 = "+ comatr.energy(probOfIJ)+'\n'+'\n');

        System.out.print("Entropy of P0,1 = "+ comatr.entropy(probOfIJ)+'\n'+'\n');

        System.out.print("Maximum probability of P0,1 = "+ comatr.MaxProbality(probOfIJ)+'\n'+'\n');

        System.out.print("Contrast of P0,1 = "+ comatr.Contrast(probOfIJ)+'\n'+'\n');

        System.out.print("Inverse difference moment of P0,1 = "+ comatr.inverseDifference(probOfIJ)+'\n'+'\n');

        System.out.print("Correlation of P0,1 = "+ comatr.correlation(probOfIJ)+'\n');


        System.out.println("Image backgroundRemoval");
        // converting back to CV_8U
        Core.convertScaleAbs( dst, abs_dst );
        HighGui.imshow( window_name, abs_dst );
        HighGui.waitKey(0);
        System.exit(0);
    }
    private static Mat doCanny(Mat frame) {
        double threshold = 0;
        // init
        Mat grayImage = new Mat();
        Mat detectedEdges = new Mat();

        // convert to grayscale
        Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

        // reduce noise with a 3x3 kernel
        Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

        // canny detector, with ratio of lower:upper threshold of 3:1
        Imgproc.Canny(detectedEdges, detectedEdges, threshold, threshold * 3, 3, false);

        // using Canny's output as a mask, display the result
        Mat dest = new Mat();
        Core.add(dest, Scalar.all(0), dest);
        frame.copyTo(dest, detectedEdges);

        return dest;
    }

    /**
     * Perform the operations needed for removing a uniform background
     *
     * @param frame the current frame
     * @return an image with only foreground objects
     */
    private static Mat doBackgroundRemoval(Mat frame) {

        // init
        Mat hsvImg = new Mat();
        List<Mat> hsvPlanes = new ArrayList<>();
        Mat thresholdImg = new Mat();

        // threshold the image with the histogram average value
        hsvImg.create(frame.size(), CvType.CV_8U);
        Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
        Core.split(hsvImg, hsvPlanes);

        double threshValue = getHistAverage(hsvImg, hsvPlanes.get(0));


        Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY_INV);

        // Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

        Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

        // dilate to fill gaps, erode to smooth edges
        Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, 1), 6);
        Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, 1), 6);

        Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

        // create the new image
        Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        frame.copyTo(foreground, thresholdImg);

        return foreground;
    }

    private static double getHistAverage(Mat hsvImg, Mat hueValues) {
        // init
        double average = 0.0;
        Mat hist_hue = new Mat();
        MatOfInt histSize = new MatOfInt(180);
        List<Mat> hue = new ArrayList<>();
        hue.add(hueValues);

        // compute the histogram
        Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

        // get the average for each bin
        for (int h = 0; h < 180; h++)
        {
            average += (hist_hue.get(h, 0)[0] * h);
        }

        return average = average / hsvImg.size().height / hsvImg.size().width;
    }
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Main lfCl=new Main();
        lfCl.run(args);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Reading the Image from the file and storing it in to a Matrix object
        String file = "C:\\Users\\KobitPC\\Desktop\\Test Pictures/orange.jpg";


    }


}
class COMatrix {

    Mat array;

    double rowsMean=0.0;
    double colsMean=0.0;

    double rowsVar=0.0;
    double colsVar=0.0;

    public COMatrix(Mat array){

        this.array=array;

    }

    public int NoOfgreylevels(){

        int max=0;

        for(int i=0;i<array.rows();i++) {
            for (int j = 0; j <array.cols(); j++) {
                double[] temp=array.get(i,j);
                if(temp[0]>max)
                    max=(int)temp[0];
            }
        }

        return max;

    }
    public int NoOfPairsOfPixelIJ(int angle, int d, int rowV, int colV){

        int pairCount=0;

        for(int i=1;i<=array.rows();i++) {
            for (int j = 0; j <array.cols(); j++) {
                if (angle==0) {
                    if(j<array.rows()-d) {
                        double[] temp=array.get(i-1,j);
                        double[] temp2=array.get(i-1,j+d);
                        if (temp[0] == rowV && temp2[0] == colV)
                            pairCount++;
                        if(temp2[0] == rowV && temp[0] == colV)
                            pairCount++;
                    }

                }

            }
        }
        return pairCount;
    }

    public double log2(double x) {
        return (Math.log(x) / Math.log(2));
    }

    public double entropy(double[][] prob){
        double entropyV=0;

        for(int i=0;i<prob.length;i++) {
            for (int j = 0; j < prob[0].length; j++) {
                if(prob[i][j]!=0.0)
                    entropyV-=prob[i][j]*log2(prob[i][j]);
            }
        }

        return entropyV;
    }
    public double Contrast(double[][] prob){

        double contrastV=0;

        for(int i=0;i<prob.length;i++) {
            for (int j = 0; j < prob[0].length; j++) {
                if(prob[i][j]!=0.0)
                    contrastV+=(((i-j)*(i-j))*prob[i][j]);
            }
        }

        return contrastV;
    }
    public void setupOfMean(double[][] arr, int sum){

        for (int i=0;i<arr.length;i++) {
            for (int j = 0; j < arr[0].length; j++) {
                rowsMean +=(double) (i)*arr[i][j];
                colsMean+=(double) (j) * arr[i][j];
            }
        }

    }
    public void setupOfVar(double[][] arr, int sum){

        for (int i=0;i<arr.length;i++) {
            for (int j = 0; j < arr[0].length; j++) {
                rowsVar+= arr[i][j]*((i-rowsMean)*(i-rowsMean));
                colsVar+= arr[i][j]*((j-colsMean)*(j-colsMean));
            }
        }

    }
    public double correlation(double[][] prob){

        double correlationV=0.0;

        for (int i=0;i<prob.length;i++) {
            for (int j = 0; j < prob[0].length; j++) {
                double den=(Math.sqrt(rowsVar*((colsVar)*(colsVar))));
                if(prob[i][j]!=0.0 && den!=0.0)
                    correlationV+=((prob[i][j]*(((i-rowsMean)*(j-colsMean))))/den);
            }
        }

        return correlationV;
    }
    public double energy(double[][] prob){

        double energyV=0.0;

        double ASM=0.0;

        for (int i=0;i<prob.length;i++) {
            for (int j = 0; j < prob[0].length; j++) {
                ASM+=prob[i][j]*prob[i][j];
            }
        }

        energyV=Math.sqrt(ASM);

        return energyV;
    }
    public double MaxProbality(double[][] prob){

        double maxP=0.0;

        for (int i=0;i<prob.length;i++) {
            for (int j = 0; j < prob[0].length; j++) {
                if (maxP < prob[i][j])
                    maxP = prob[i][j];
            }
        }

        return maxP;
    }
    public double inverseDifference(double[][] prob){

        double inverseD=0.0;

        for (int i=0;i<prob.length;i++) {
            for (int j = 0; j < prob[0].length; j++) {
                inverseD+=(double) (1/(1+((i-j)*(i-j))))*prob[i][j];
            }
        }

        return inverseD;
    }

}
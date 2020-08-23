package com.example.catpredictor;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private Button btnCapture;
    private ImageView imgCapture;
    private TextView textMessage;

    private static final int Image_Capture_Code = 1;
    String mCurrentPhotoPath;
    Bitmap bitmap;
    float probability;
    String text;
    Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btnCapture =(Button)findViewById(R.id.btnTakePicture);
        imgCapture = (ImageView) findViewById(R.id.capturedImage);

        btnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dispatchTakePictureIntent();
                galleryAddPic();
            }
        });

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == Image_Capture_Code) {
            if (resultCode == RESULT_OK) {
                Toast.makeText(this, "Photo saved successfully", Toast.LENGTH_LONG).show();
                setPic();
                PredictCat();

                text = "cat probability " + String.format("%.1f", probability*100) + "%";
                textMessage = (TextView)findViewById(R.id.textView);
                textMessage.setBackgroundColor(Color.parseColor("#70C3C1C1"));
                textMessage.setText(text);

            } else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
                Toast.makeText(this, "CError occurred while creating the File", Toast.LENGTH_LONG).show();
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this,
                        "com.example.android.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, Image_Capture_Code);
            }
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);

        // Save a file: path for use with ACTION_VIEW intents
        mCurrentPhotoPath = image.getAbsolutePath();
        Log.i("mCurrentPhotoPath", mCurrentPhotoPath);
        return image;
    }

    private void galleryAddPic() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(mCurrentPhotoPath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        this.sendBroadcast(mediaScanIntent);
    }

    private void setPic() {
        Log.i("setPic", "begin");

        // Get the dimensions of the View
        int targetW = imgCapture.getWidth();
        int targetH = imgCapture.getHeight();

        // Get the dimensions of the bitmap
        BitmapFactory.Options bmOptions = new BitmapFactory.Options();
        bmOptions.inJustDecodeBounds = true;

        BitmapFactory.decodeFile(mCurrentPhotoPath, bmOptions);

        int photoW = bmOptions.outWidth;
        int photoH = bmOptions.outHeight;

        // Determine how much to scale down the image
        int scaleFactor = Math.max(1, Math.min(photoW/targetW, photoH/targetH));

        // Decode the image file into a Bitmap sized to fill the View
        bmOptions.inJustDecodeBounds = false;
        bmOptions.inSampleSize = scaleFactor;
        bmOptions.inPurgeable = true;

        bitmap = BitmapFactory.decodeFile(mCurrentPhotoPath, bmOptions);
        imgCapture.setImageBitmap(bitmap);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("converted_main_model.tflite");
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffSets = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffSets, declaredLength);
    }

    private void PredictCat() {
        probability = 0;

        // Preprocess image for prediction
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true);

        float pixel_sum = 0;
        for (int y = 0; y < scaledBitmap.getHeight(); y++) {
            for (int x = 0; x < scaledBitmap.getWidth(); x++) {
                int c1 = scaledBitmap.getPixel(x, y);
                pixel_sum += Color.red(c1) + Color.green(c1) + Color.blue(c1);
            }
        }
        float image_mean = pixel_sum/(49152);

        float pixel_dev_sum = 0;
        for (int y = 0; y < scaledBitmap.getHeight(); y++) {
            for (int x = 0; x < scaledBitmap.getWidth(); x++) {
                int c2 = scaledBitmap.getPixel(x, y);
                pixel_dev_sum += (Color.red(c2)-image_mean)*(Color.red(c2)-image_mean) +
                        (Color.green(c2)-image_mean)*(Color.green(c2)-image_mean) +
                        (Color.blue(c2)-image_mean)*(Color.blue(c2)-image_mean);
            }
        }
        float image_std = (float) Math.sqrt(pixel_dev_sum/49152);

        // send image to model using byte buffer
        try {
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 128 * 128 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[128 * 128];
            scaledBitmap.getPixels(intValues, 0, 128, 0, 0, 128, 128);

            int pixel = 0;
            for (int y = 0; y < 128; y++) {
                for (int x = 0; x < 128; x++) {
                    final int val = intValues[pixel++];
                    byteBuffer.putFloat((((val >> 16) & 0xFF)-image_mean)/image_std);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)-image_mean)/image_std);
                    byteBuffer.putFloat((((val) & 0xFF)-image_mean)/image_std);
                }
            }

            tflite = new Interpreter(loadModelFile());
            int[] dims = {1, 128, 128, 3};
            tflite.resizeInput(0, dims);
            float[][] model_output = new float[1][1];
            tflite.run(byteBuffer, model_output);
            probability = model_output[0][0];
            Log.i("probability", Float.toString(probability));

        } catch (Exception e) {
            Log.e("Prediction Error", Log.getStackTraceString(e));
        }
    }
}

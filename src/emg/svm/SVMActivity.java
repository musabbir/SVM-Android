package emg.svm;

import java.io.IOException;
import java.util.ArrayList;
import android.util.Log;
import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import libsvm.*;
import emg.backend.*;
//import emg.bluetooth.test;


public class SVMActivity extends Activity implements  ITransformListener{
    /** Called when the activity is first created. */
	private static final String TAG = "Libsvm";
	byte[] y;
    double Cp, Cn;
    String DEVICE_ADDRESS = "00:06:66:04:9B:21";
	byte channels = 1;//each bit is a channel (bits1-6)
    DataProtocol dataProtocol;
    
    ArrayList<nodeData> nodeContainer;
    long Delay = 5000;
    
    svm_problem prob;
    TextView resultText;
    TextView tv;
    View view1;
    Button trainButton;
    public boolean trainingstate = false;
    public boolean testingstate = false;
    
   
    svm_model initialmodel;
    public boolean trainingComplete = false;
    long time;


    // has a touch press started?
    public boolean touchStarted = false;
    public boolean trainingInProgress = false;
             
       
    
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        this.trainButton = (Button)findViewById(R.id.button1);
        
        this.nodeContainer = new ArrayList();      
        this.view1 = (View)findViewById(R.id.view1);
        this.resultText = (TextView) findViewById(R.id.result_text); 
        this.resultText.setText("onCreate()");
        
        //targetView = (View)findViewById(R.id.mainlayout);
        view1.setOnTouchListener(new View.OnTouchListener() {
			
        	public boolean onTouch(View v, MotionEvent event)
            {
             int action = event.getAction();
             if (action == MotionEvent.ACTION_DOWN)
             {
              touchStarted = true;
              //Log.d(TAG, "ActionDown Detected" );
              view1.setBackgroundColor(0x0000FF00);
             }             
             else if (action == MotionEvent.ACTION_UP)
             {
            	 touchStarted = false;
            	// Log.d(TAG, "ActionUp Detected" );
            	 view1.setBackgroundColor(0xFFFF0000);
            	 
             }

             return touchStarted;
            }
        });
        //this will start the training sequence.
        trainButton.setOnClickListener(new View.OnClickListener() {			
			public void onClick(View v) {
				train();				
			}
		});
        //start classifying on the input from bluetooth
        
    }
    
    @Override
    public void onDestroy()
    {
    	this.dataProtocol.StopAndDisconnect();
    	super.onPause();
    }
    
    //this class contains
    private class nodeData{
    	public int[] data;
    	public double output;
    	
    	public nodeData(int[] data, double output){
    		this.data = data;
    		this.output = output;
    	}
    	
    }
    
    
    //reading training data from blue-tooth
    //put training data into nodeData class and add it to 

    //1.0 or 0.0 is passed into nodeData depending on whether screentouch is true or false
   
    public void addData(int[] data) {
    		long timenow = System.currentTimeMillis();
    		if ((timenow - time) > Delay) trainingstate = false;
    		
	    	nodeData temp;
	    	if(touchStarted){    		
	    		temp = new nodeData(data, 1.0);
	    		if (trainingstate) nodeContainer.add(temp);
	    		//Log.d(TAG, "ActionDown add" );
	    	}else{
	    		temp = new nodeData(data, 0.0);
	    		if (trainingstate) nodeContainer.add(temp);
	    		//Log.d(TAG, "action up add" );
	    		//System.out.println( 0);
	    	}    	    		
	    	//Log.d(TAG, "Length: " + Integer.toString(nodeContainer.size()));
	    	if (!trainingstate){
		    	if(!trainingComplete){	
		    		synchronized (this) {	
			    		if(!trainingInProgress){
			    			
					    	Log.e(TAG, "about to try to make model" );
				    		trainingInProgress = true;
				    		initialmodel = makeModel();
					        
					        Log.e(TAG, "Model construction complete" );
					        trainingComplete = true;
			    		}
		    		}
		    	}
	    	
	    	
		    	if (trainingComplete) predict_results(initialmodel, temp);
	    	}//Log.d(TAG, "Prediction thread" );
    	}
    	
	
    
    public svm_model makeModel(){
    	svm_node node;
    	prob = new svm_problem();
    	svm_model model;
    	svm_parameter subparam = new svm_parameter();
		subparam.svm_type = svm_parameter.C_SVC;
		subparam.kernel_type = svm_parameter.LINEAR;
		subparam.degree = 3;
		subparam.gamma = 0;
		subparam.coef0 = 0;
		subparam.nu = 0.5;
		subparam.cache_size = 100;
		subparam.C = 0.1;
		subparam.eps = 0.001;
		subparam.p = 0.1;
		subparam.shrinking = 1;
		subparam.probability = 0;
		subparam.nr_weight = 0;
		subparam.weight_label = new int[2];
		subparam.weight = new double[2];
		subparam.weight[0] = 0;
		subparam.weight[1] = 0;
    	
		Log.d(TAG, "Making model" );
    	int length = nodeContainer.size();
    
    	prob.x = new svm_node[length][8];
    	prob.y = new double[length];
		
		prob.l = length;
		double output;
		Log.d(TAG, "Prob.l x and y defined" );
		Log.d(TAG, "Length: " + Integer.toString(length));
    	for (int i=0; i < length; i++){
    		
    		output = nodeContainer.get(i).output;
    		prob.y[i] = output;
    		//Log.d(TAG, "prob.y set");
    		int count =0;
    		for (int j=0;j<nodeContainer.get(i).data.length;j++)
    		{    	
    			prob.x[i][j] = new svm_node();
    			//Log.d(TAG, "index setting" );
	    		prob.x[i][j].index = count;
	    		//Log.d(TAG, "index set" );
	    		prob.x[i][j].value = (double) nodeContainer.get(i).data[j]; 
	    		count++;
    		}
    		//Log.d(TAG, "trying to build svmnode" );
    	}Log.d(TAG, "SVM problem defined" );
    	model = svm.svm_train(prob, subparam);
    	Log.d(TAG, "Model created" );
    	return model;
    	
    }    
    public void train(){
    	Log.d(TAG, "train button clicked" );
    	tv = (TextView) findViewById(R.id.start_train_text); 
        tv.setText("Start Training");
        //setContentView(R.layout.main);
        trainingstate = true;
    	this.dataProtocol = new DataProtocol(this, (ITransformListener) this, channels, DEVICE_ADDRESS);
        this.dataProtocol.Start();
       
        time = System.currentTimeMillis();
        //SystemClock.sleep(Delay);
        
        
     
        
        //trainingstate = false;
         	
    }
    
    public void predict_results(svm_model initialmodel, nodeData node){

    	//Log.d(TAG, "prediction thread" );
    		
    		//nodeData data = nodeContainer.get(nodeContainer.size() -1);
        	svm_node[] currentNode = new svm_node[8];
        	for (int i = 0; i < 8; i++){
        		currentNode[i] = new svm_node();
        		currentNode[i].index = i;
        		currentNode[i].value = node.data[i];
        	}

    	double result = svm.svm_predict(initialmodel, currentNode);
    	Message msg = new Message();
    	msg.obj = result;
    	this.newDataHandler.sendMessage(msg);
    }
    
    Handler newDataHandler = new Handler(){
		@Override
		public void handleMessage(Message msg){
			double result = (Double) msg.obj;
			
			String s = "";
			
			s = Double.toString(result);
			
			resultText.setText(s);
			if (result == 1.0){
				view1.setBackgroundColor(0xFFFF0000);
			}else view1.setBackgroundColor(0x0000FF00);
			//Log.d(TAG, "end of handler()");
		}
	};
    
        	
	
        
    	
    
    
    
    
    
}
    
    
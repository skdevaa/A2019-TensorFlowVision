/*
 * Copyright (c) 2019 Automation Anywhere.
 * All rights reserved.
 *
 * This software is the proprietary information of Automation Anywhere.
 * You shall use it only in accordance with the terms of the license agreement
 * you entered into with Automation Anywhere.
 */

package com.automationanywhere.botcommand.sk;

import static com.automationanywhere.commandsdk.model.AttributeType.TEXT;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import com.automationanywhere.botcommand.data.Value;
import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.NumberValue;
import com.automationanywhere.commandsdk.annotations.BotCommand;
import com.automationanywhere.commandsdk.annotations.CommandPkg;
import com.automationanywhere.commandsdk.annotations.Execute;
import com.automationanywhere.commandsdk.annotations.Idx;
import com.automationanywhere.commandsdk.annotations.Pkg;
import com.automationanywhere.commandsdk.annotations.Sessions;
import com.automationanywhere.commandsdk.annotations.rules.NotEmpty;
import com.automationanywhere.commandsdk.model.AttributeType;
import com.automationanywhere.commandsdk.model.DataType;

import tsutils.DetectedObj;
import tsutils.TensorFlowSession;
import tsutils.Utils;





/**
 * @author Stefan Karsten
 *
 */

@BotCommand
@CommandPkg(label="Detect Objects", name="Detect Objects", description="Detect Objects",  icon="",
node_label="Detect Objects" ,
return_type=DataType.DICTIONARY, return_sub_type=DataType.NUMBER, return_label="Objects", return_required=true)
public class DetectObjects {
	
	private static final Logger logger = LogManager.getLogger(DetectObjects.class);
	
    @Sessions
    private Map<String, Object> sessions;
	
	@Execute
	public DictionaryValue action(@Idx(index = "1", type = TEXT) @Pkg(label = "Session name", default_value_type = STRING,  default_value = "Default") @NotEmpty String sessionName,
				       @Idx(index = "2", type = AttributeType.FILE) @Pkg(label = "Image", default_value_type = DataType.FILE) @NotEmpty String imagefile,
				       @Idx(index = "3", type = AttributeType.FILE) @Pkg(label = "Image with Detections", default_value_type = DataType.FILE) @NotEmpty String imagedeteced
						) throws Exception {
		
		TensorFlowSession tssession  = (TensorFlowSession) this.sessions.get(sessionName); 
    	SavedModelBundle model  = tssession.getModel();
    	
    	Image image = ImageIO.read(new File(imagefile));
    	BufferedImage bufferimg = (BufferedImage) image;
        Tensor<UInt8> input = Utils.makeImageTensor(bufferimg);
        List<Tensor<?>> outputs = model
		        .session()
		        .runner()
		        .feed("image_tensor", input)
		        .fetch("detection_scores")
		        .fetch("detection_classes")
		        .fetch("detection_boxes")
		        .run();
  
        
  
    
        List<DetectedObj> result = Utils.getDetections(outputs,tssession.getLabels());
    
        if (imagedeteced != null) {
        	BufferedImage imagenew = Utils.drawDetectedObjects(bufferimg, result);
        	File outputfile = new File(imagedeteced);
        	ImageIO.write(imagenew, "jpg", outputfile);
    	}
        
    	HashMap<String,Value> map = new HashMap();
    	int index = 1;
    	for (DetectedObj detectedObj : result) {
    		 map.put(detectedObj.getLabel()+"_"+Integer.toString(index++), new NumberValue(detectedObj.getScore()));
		}
	   
	    DictionaryValue dictmap = new DictionaryValue();
	    dictmap.set(map);
	    return dictmap;

	}
	
	
    public void setSessions(Map<String, Object> sessions) {
        this.sessions = sessions;
    }
 
}

	



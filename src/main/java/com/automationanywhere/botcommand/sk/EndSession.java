/*
 * Copyright (c) 2019 Automation Anywhere.
 * All rights reserved.
 *
 * This software is the proprietary information of Automation Anywhere.
 * You shall use it only in accordance with the terms of the license agreement
 * you entered into with Automation Anywhere.
 */
/**
 * 
 */
package com.automationanywhere.botcommand.sk;

import static com.automationanywhere.commandsdk.model.AttributeType.TEXT;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

import java.io.File;
import java.nio.file.Paths;
import java.util.Map;

import org.apache.logging.log4j.core.util.FileUtils;
import org.tensorflow.SavedModelBundle;

import com.automationanywhere.commandsdk.annotations.BotCommand;
import com.automationanywhere.commandsdk.annotations.CommandPkg;
import com.automationanywhere.commandsdk.annotations.Execute;
import com.automationanywhere.commandsdk.annotations.Idx;
import com.automationanywhere.commandsdk.annotations.Pkg;
import com.automationanywhere.commandsdk.annotations.Sessions;
import com.automationanywhere.commandsdk.annotations.rules.NotEmpty;

import tsutils.TensorFlowSession;



/**
 * @author Stefan Karsten
 *
 */

@BotCommand
@CommandPkg(label = "End session", name = "EndSession", description = "End session", icon = 
"", node_label = "End session {{sessionName}}")
public class EndSession {


    @Sessions
    private Map<String, Object> sessions;
 
    @Execute
    public void end(
            @Idx(index = "1", type = TEXT) @Pkg(label = "Session name", default_value_type = STRING, 
            default_value = "Default") @NotEmpty String sessionName) {
    	
    	TensorFlowSession  tssession  = (TensorFlowSession) this.sessions.get(sessionName);  
    	tssession.getModel().close();
    	new File(tssession.getTempDir()+"/labels.txt").delete();  
    	new File(tssession.getTempDir()+"/saved_model.pb").delete();   
    	new File(tssession.getTempDir()).delete(); 

        sessions.remove(sessionName);
         
    }
     
    public void setSessions(Map<String, Object> sessions) {
        this.sessions = sessions;
    }
}
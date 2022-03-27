package com.deep.server;

import com.deep.server.framework.DeepServer;
import org.junit.Test;

public class AppTest {

    @Test
    public void appTest() throws Exception {
        DeepServer.run(6666);
    }

}

package com.deep.framework;

import com.deep.framework.ast.Parser;
import org.junit.Test;

public class ASTTest {

    @Test
    public void arrayTest() {
        Parser parser = new Parser();
        parser.parser("/Users/chengdong/GitHub/deep-learning/deep-framework/src/main/java/com/deep/framework/graph/TensorOparetor.java");
    }

}

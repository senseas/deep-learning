package com.deep.framework;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.statement.BlockStatement;
import com.deep.framework.jit.statement.ImportStatement;
import com.deep.framework.jit.statement.PackageStatement;
import com.deep.framework.jit.statement.Statement;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class ASTTest {


    @Test
    public void arrayTest() {
        Parser parser = new Parser();
        parser.parser("/Users/chengdong/GitHub/deep-learning/deep-framework/src/main/java/com/deep/framework/graph/Tensor.java");
        parser.parser(null,null,parser.list);
    }

}

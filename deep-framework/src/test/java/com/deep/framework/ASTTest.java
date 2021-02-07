package com.deep.framework;

import com.deep.framework.jit.Parser;
import org.junit.Test;

public class ASTTest {

    @Test
    public void arrayTest() {
        Parser parser = new Parser();
        parser.parser("D:\\GitHub\\deep-learning\\deep-framework\\src\\main\\java\\com\\deep\\framework\\graph\\Tensor.java");
    }

}

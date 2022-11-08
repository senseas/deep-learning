package com.deep.framework;

import com.deep.framework.ast.Parser;
import org.junit.Test;

public class ASTTest {

    @Test
    public void arrayTest() {
        Parser parser = new Parser();
        parser.parser("D:\\github\\deep-learning\\deep-framework\\src\\main\\java\\com\\deep\\framework\\graph\\Tensor.java");
    }

}

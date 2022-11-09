
package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.statement.Statement;

public class LambdaExpression extends Expression {

    public ParametersExpression parameters;
    private Statement body;

    public LambdaExpression(Node prarent) {
        super(prarent);
    }
}

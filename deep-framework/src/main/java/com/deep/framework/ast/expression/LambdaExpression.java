
package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.statement.Statement;

public class LambdaExpression extends Expression {

    private Statement body;

    public LambdaExpression(Node prarent) {
        super(prarent);
    }
}

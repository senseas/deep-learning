package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;

import static com.deep.framework.ast.lexer.TokenType.WHILE;

public class WhileStatement extends Statement {
    private Expression condition;
    private BlockStatement body;

    public void parser(Node node) {
        if (node.getChildrens().contains(WHILE)) {
            int index = node.getChildrens().indexOf(WHILE);
            Object a = node.getChildrens().get(index + 1);
            Object b = node.getChildrens().get(index + 2);
            condition = (ParametersExpression) a;
            body = (BlockStatement) b;
        }
    }
}

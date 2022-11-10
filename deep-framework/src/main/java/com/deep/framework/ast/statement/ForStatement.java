package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.IF;

public class ForStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;

    public ParametersExpression parameters;
    private BlockStatement body;

    public void parser(Node node) {
        if (node.getChildrens().contains(IF)) {
            int index = node.getChildrens().indexOf(IF);
            Object a = node.getChildrens().get(index + 1);
            Object b = node.getChildrens().get(index + 2);
            parameters = (ParametersExpression) a;
            body = (BlockStatement) b;
        }
    }
}
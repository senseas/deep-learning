package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;

import static com.deep.framework.ast.lexer.TokenType.FOR;
import static com.deep.framework.ast.lexer.TokenType.WHILE;

public class WhileStatement extends Statement {
    private Expression condition;
    public ParametersExpression parameters;
    private BlockStatement body;

    public void parser(Node node) {
        if (node.getChildrens().contains(WHILE)) {
            setPrarent(node.getPrarent());
            int index = node.getChildrens().indexOf(WHILE);
            Object a = node.getChildrens().get(index + 1);
            parameters = (ParametersExpression) a;
            int index1 = node.getPrarent().getChildrens().indexOf(node);
            Object b = node.getPrarent().getChildrens().get(index1 + 1);
            body = (BlockStatement) b;
            node.getChildrens().add(b);
            node.getPrarent().getChildrens().set(index1, this);
            node.getPrarent().getChildrens().remove(b);
            node.getChildrens().remove(WHILE);
            this.setChildrens(node.getChildrens());
        }
    }
}

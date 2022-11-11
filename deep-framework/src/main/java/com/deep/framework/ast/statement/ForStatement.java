package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.FOR;

public class ForStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;

    public ParametersExpression parameters;
    private BlockStatement body;

    public void parser(Node node) {
        if (node.getChildrens().contains(FOR)) {
            setPrarent(node.getPrarent());
            int index = node.getChildrens().indexOf(FOR);
            Object a = node.getChildrens().get(index + 1);
            parameters = (ParametersExpression) a;
            int index1 = node.getPrarent().getChildrens().indexOf(node);
            Object b = node.getPrarent().getChildrens().get(index1 + 1);
            body = (BlockStatement) b;
            node.getChildrens().add(b);
            node.getPrarent().getChildrens().set(index1, this);
            node.getPrarent().getChildrens().remove(b);
            node.getChildrens().remove(FOR);
            this.setChildrens(node.getChildrens());
        }
    }
}
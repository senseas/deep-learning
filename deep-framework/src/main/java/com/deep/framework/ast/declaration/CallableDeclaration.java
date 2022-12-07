package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.LambdaExpression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

@Data
public class CallableDeclaration extends Declaration {
    private Expression expression;
    private Expression parameters;

    public CallableDeclaration(Node prarent, Expression expression, Expression parameters) {
        super(prarent);
        this.expression = expression;
        this.parameters = parameters;

        this.expression.setPrarent(this);
        this.parameters.setPrarent(this);

        getChildrens().addAll(expression, parameters);
    }

    public static void parser(Node node) {
        if (node instanceof CallableDeclaration) return;
        if (node instanceof MethodDeclaration) return;
        LambdaExpression.parser(node);
        Stream.of(node.getChildrens()).reduce2((list, m, n) -> {
            if (m instanceof Expression && n instanceof ParametersExpression) {
                CallableDeclaration declare = new CallableDeclaration(node, (Expression) m, (Expression) n);
                list.replace(m, declare);
                list.remove(n);
            }
        });
    }

    @Override
    public String toString() {
        return expression.toString().concat(parameters.toString());
    }
}
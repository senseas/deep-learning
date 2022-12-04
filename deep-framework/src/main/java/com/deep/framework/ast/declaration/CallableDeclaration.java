package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.LambdaExpression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

@Data
public class CallableDeclaration extends Declaration {
    private Expression parameters;
    private Name name;

    public CallableDeclaration(Node prarent, Name name, Expression parameters) {
        super(prarent);
        this.parameters = parameters;
        this.name = name;

        this.name.setPrarent(this);
        this.parameters.setPrarent(this);

        getChildrens().addAll(name, parameters);
    }

    public static void parser(Node node) {
        if (node instanceof CallableDeclaration) return;
        if (node instanceof MethodDeclaration) return;
        LambdaExpression.parser(node);
        Stream.of(node.getChildrens()).reduce2((list, m, n) -> {
            if (m instanceof Name && n instanceof ParametersExpression) {
                CallableDeclaration declare = new CallableDeclaration(node, (Name) m, (Expression) n);
                list.replace(m, declare);
                list.remove(n);
            }
        });
    }

    @Override
    public String toString() {
        return name.toString().concat(parameters.toString());
    }
}
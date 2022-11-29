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
    private static CallableDeclaration declare;

    public CallableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof CallableDeclaration) return;
        if (node instanceof MethodDeclaration) return;
        LambdaExpression.parser(node);
        Stream.of(node.getChildrens()).reduce2((list, m, n) -> {
            if (m instanceof Name && n instanceof ParametersExpression) {
                declare = new CallableDeclaration(node);
                declare.setName((Name) m);
                declare.setParameters((Expression) n);
                declare.getChildrens().addAll(m, n);
                m.setPrarent(declare);
                n.setPrarent(declare);
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
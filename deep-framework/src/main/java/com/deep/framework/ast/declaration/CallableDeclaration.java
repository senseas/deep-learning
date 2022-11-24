package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;


@Data
public class CallableDeclaration extends Declaration {
    private ParametersExpression parameters;
    private Name name;
    private static CallableDeclaration declare;

    public CallableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof CallableDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (m instanceof Name && n instanceof ParametersExpression) {
                declare = new CallableDeclaration(node);
                declare.setName((Name) m);
                declare.setParameters((ParametersExpression) n);
                declare.getChildrens().add(n);
                node.replaceAndRemove(m, declare, n);
                list.remove(n);
            }
        });
    }

    @Override
    public String toString() {
        return name.toString().concat("(").concat(parameters.toString().concat(")"));
    }
}

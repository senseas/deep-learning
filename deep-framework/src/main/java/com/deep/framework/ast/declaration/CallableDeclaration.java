package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;


@Data
public class CallableDeclaration extends Declaration {
    private ParametersExpression parameters;
    private Name name;
    private static CallableDeclaration callable;

    public CallableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof CallableDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (m instanceof Name && n instanceof ParametersExpression) {
                callable = new CallableDeclaration(node);
                callable.setName((Name) m);
                callable.setParameters((ParametersExpression) n);
                callable.getChildrens().add(n);
                node.replaceAndRemove(m, callable, n);
                list.remove(n);
            }
        });
    }

    @Override
    public String toString() {
        return name.toString().concat("(").concat(parameters.toString().concat(")"));
    }
}

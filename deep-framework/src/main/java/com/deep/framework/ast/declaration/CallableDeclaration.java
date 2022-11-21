package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;

import java.util.List;

public class CallableDeclaration extends Declaration {
    private ParametersExpression parameters;
    private Name name;
    private static CallableDeclaration callable;

    public CallableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (m instanceof Name a) {
                if (n instanceof ParametersExpression b) {
                    callable = new CallableDeclaration(node);
                    callable.setName(a);
                    callable.setParameters(b);
                    callable.setChildrens(a.getChildrens());
                    callable.getChildrens().add(n);
                    node.replaceAndRemove(a, callable, n);
                    list.remove(n);
                }
            }
        });
    }

    public void setName(Name name) {
        this.name = name;
    }

    public void setParameters(ParametersExpression parameters) {
        this.parameters = parameters;
    }
}

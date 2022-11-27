package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.CATCH;

@Data
public class CatchClause extends Statement {
    public ParametersExpression parameters;
    private Statement body;

    public static void parser(Node node) {
        if (node instanceof CatchClause) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(CATCH) && n instanceof ParametersExpression) {
                    if (b instanceof BlockStatement) {
                        //create CatchClause and set Prarent，Parameters
                        CatchClause statement = new CatchClause();
                        statement.setPrarent(node);
                        statement.setParameters((ParametersExpression) n);
                        statement.setBody((BlockStatement) b);
                        statement.getChildrens().addAll(List.of(n, b));

                        //remove CatchNode and Parameters
                        b.setPrarent(statement);
                        a.getChildrens().clear();
                        node.replaceAndRemove(a, statement, b);
                        list.remove(b);
                    } else {
                        throw new RuntimeException("CatchClause error ".concat(node.toString()));
                    }
                }
            });
        });
    }
}